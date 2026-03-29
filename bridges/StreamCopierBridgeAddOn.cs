#region Using declarations
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Core;
using NinjaTrader.NinjaScript;
#endregion

// Place this file at:
// Documents\NinjaTrader 8\bin\Custom\AddOns\StreamCopierBridgeAddOn.cs
// Then compile from NinjaTrader (New -> NinjaScript Editor -> Compile).

namespace NinjaTrader.NinjaScript.AddOns
{
    public class StreamCopierBridgeAddOn : AddOnBase
    {
        // ===== Bridge config =====
        // Use the same URL + token in backend/.env:
        // NINJATRADER_BRIDGE_URL=http://127.0.0.1:18080
        // NINJATRADER_BRIDGE_TOKEN=<same token>
        // For WSL backend connectivity, bind to all local interfaces (not just 127.0.0.1).
        // Keep RequiredBearerToken set in production.
        private const string ListenerPrefix = "http://*:18080/";
        private const string RequiredBearerToken = "";
        private const string DefaultAccountName = ""; // e.g. "Sim101" or your live account name
        private const int DispatchTimeoutMs = 5000;
        private const double FallbackWideStopPoints = 120.0;
        private const double FallbackWideTargetPoints = 240.0;

        private readonly object sync = new object();
        // Pending protection intents keyed by account|instrument. We apply them when
        // a non-flat position update confirms the entry/add fill.
        private readonly Dictionary<string, PendingProtection> pendingProtections = new Dictionary<string, PendingProtection>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, ProtectionState> protectionBySymbol = new Dictionary<string, ProtectionState>(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> subscribedAccounts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        private HttpListener listener;
        private CancellationTokenSource listenerCts;
        private Task listenerTask;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "StreamCopierBridgeAddOn";
                return;
            }

            if (State == State.Active)
            {
                StartBridge();
                return;
            }

            if (State == State.Terminated)
            {
                StopBridge();
            }
        }

        private void StartBridge()
        {
            if (listenerTask != null)
                return;

            try
            {
                listenerCts = new CancellationTokenSource();
                listener = new HttpListener();
                listener.Prefixes.Add(EnsureTrailingSlash(ListenerPrefix));
                listener.Start();
                listenerTask = Task.Run(() => ListenLoopAsync(listenerCts.Token));
                Print("[StreamCopierBridge] Listener started on " + ListenerPrefix);
            }
            catch (Exception ex)
            {
                Print("[StreamCopierBridge] Failed to start listener: " + ex);
                Print("[StreamCopierBridge] If needed, run as admin: netsh http add urlacl url=http://127.0.0.1:18080/ user=%USERNAME%");
            }
        }

        private void StopBridge()
        {
            try
            {
                if (listenerCts != null && !listenerCts.IsCancellationRequested)
                    listenerCts.Cancel();
            }
            catch
            {
                // ignored on shutdown
            }

            try
            {
                if (listener != null && listener.IsListening)
                    listener.Stop();
            }
            catch
            {
                // ignored on shutdown
            }

            try
            {
                if (listener != null)
                    listener.Close();
            }
            catch
            {
                // ignored on shutdown
            }

            listener = null;
            listenerTask = null;
            listenerCts = null;
        }

        private async Task ListenLoopAsync(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                HttpListenerContext context = null;
                try
                {
                    context = await listener.GetContextAsync().ConfigureAwait(false);
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (HttpListenerException)
                {
                    if (token.IsCancellationRequested)
                        return;
                    continue;
                }
                catch (InvalidOperationException)
                {
                    if (token.IsCancellationRequested)
                        return;
                    continue;
                }

                _ = Task.Run(() => HandleRequestAsync(context), token);
            }
        }

        private async Task HandleRequestAsync(HttpListenerContext context)
        {
            try
            {
                string path = context.Request.Url != null
                    ? context.Request.Url.AbsolutePath.TrimEnd('/').ToLowerInvariant()
                    : string.Empty;

                if (context.Request.HttpMethod == "GET" && path == "/api/stream-copier/health")
                {
                    await WriteJsonAsync(context, 200, BuildHealthResponse()).ConfigureAwait(false);
                    return;
                }

                if (context.Request.HttpMethod == "GET" && path == "/api/stream-copier/state")
                {
                    if (!IsAuthorized(context.Request))
                    {
                        await WriteJsonAsync(context, 401, BridgeResponse.Error("unauthorized", "Missing or invalid bearer token.")).ConfigureAwait(false);
                        return;
                    }

                    string accountName = context.Request.QueryString != null ? context.Request.QueryString["account"] : null;
                    string symbol = context.Request.QueryString != null ? context.Request.QueryString["symbol"] : null;
                    var result = DispatchStateRequest(accountName, symbol);
                    int status = result.Ok ? 200 : 400;
                    await WriteJsonAsync(context, status, result).ConfigureAwait(false);
                    return;
                }

                if (context.Request.HttpMethod == "POST" && path == "/api/stream-copier/commands")
                {
                    if (!IsAuthorized(context.Request))
                    {
                        await WriteJsonAsync(context, 401, BridgeResponse.Error("unauthorized", "Missing or invalid bearer token.")).ConfigureAwait(false);
                        return;
                    }

                    string body;
                    using (var reader = new StreamReader(context.Request.InputStream, context.Request.ContentEncoding ?? Encoding.UTF8))
                    {
                        body = await reader.ReadToEndAsync().ConfigureAwait(false);
                    }

                    BridgeCommand command = null;
                    try
                    {
                        command = DeserializeCommand(body);
                    }
                    catch (Exception ex)
                    {
                        await WriteJsonAsync(context, 400, BridgeResponse.Error("invalid_json", ex.Message)).ConfigureAwait(false);
                        return;
                    }

                    if (command == null)
                    {
                        await WriteJsonAsync(context, 400, BridgeResponse.Error("invalid_command", "Empty command payload.")).ConfigureAwait(false);
                        return;
                    }

                    var result = DispatchCommand(command);
                    int status = result.Ok ? 200 : 400;
                    await WriteJsonAsync(context, status, result).ConfigureAwait(false);
                    return;
                }

                await WriteJsonAsync(context, 404, BridgeResponse.Error("not_found", "Route not found.")).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                await WriteJsonAsync(context, 500, BridgeResponse.Error("bridge_error", ex.ToString())).ConfigureAwait(false);
            }
        }

        private BridgeResponse DispatchCommand(BridgeCommand command)
        {
            var done = new ManualResetEventSlim(false);
            BridgeResponse result = null;

            Action run = () =>
            {
                try
                {
                    result = ExecuteCommand(command);
                }
                catch (Exception ex)
                {
                    result = BridgeResponse.Error("execution_error", ex.ToString());
                }
                finally
                {
                    done.Set();
                }
            };

            if (Globals.RandomDispatcher != null)
                Globals.RandomDispatcher.BeginInvoke(run, DispatcherPriority.Normal);
            else
                run();

            if (!done.Wait(DispatchTimeoutMs))
                return BridgeResponse.Error("dispatch_timeout", "NinjaTrader command dispatch timed out.");

            return result ?? BridgeResponse.Error("dispatch_error", "Unknown command dispatch failure.");
        }

        private BridgeResponse DispatchStateRequest(string accountName, string symbol)
        {
            var done = new ManualResetEventSlim(false);
            BridgeResponse result = null;

            Action run = () =>
            {
                try
                {
                    result = BuildStateResponse(accountName, symbol);
                }
                catch (Exception ex)
                {
                    result = BridgeResponse.Error("state_error", ex.ToString());
                }
                finally
                {
                    done.Set();
                }
            };

            if (Globals.RandomDispatcher != null)
                Globals.RandomDispatcher.BeginInvoke(run, DispatcherPriority.Normal);
            else
                run();

            if (!done.Wait(DispatchTimeoutMs))
                return BridgeResponse.Error("dispatch_timeout", "NinjaTrader state dispatch timed out.");

            return result ?? BridgeResponse.Error("dispatch_error", "Unknown state dispatch failure.");
        }

        private BridgeResponse BuildStateResponse(string accountName, string symbol)
        {
            string resolvedAccountName = !string.IsNullOrWhiteSpace(accountName) ? accountName : DefaultAccountName;
            Account account = ResolveAccount(resolvedAccountName);
            if (account == null)
                return BridgeResponse.Error("account_not_found", "Account not found: " + resolvedAccountName);

            SubscribeAccountEvents(account);

            Instrument instrument = null;
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                instrument = Instrument.GetInstrument(symbol);
                if (instrument == null)
                    return BridgeResponse.Error("instrument_not_found", "Instrument not found: " + symbol + ". Use exact NinjaTrader instrument (e.g. NQ 06-26).");
            }
            else
            {
                instrument = FindFirstOpenInstrument(account);
            }

            Position position = instrument != null ? FindPosition(account, instrument) : null;
            bool hasPosition = position != null && position.MarketPosition != MarketPosition.Flat && Math.Abs(position.Quantity) > 0;
            string marketPosition = hasPosition
                ? (position.MarketPosition == MarketPosition.Long ? "LONG" : "SHORT")
                : "FLAT";

            ProtectionState trackedProtection = null;
            if (instrument != null)
            {
                string protectionKey = BuildProtectionKey(account, instrument);
                lock (sync)
                {
                    protectionBySymbol.TryGetValue(protectionKey, out trackedProtection);
                }
            }

            double? stopPrice = trackedProtection != null ? trackedProtection.StopPrice : null;
            double? targetPrice = trackedProtection != null ? trackedProtection.TargetPrice : null;
            if (instrument != null && hasPosition && (!stopPrice.HasValue || !targetPrice.HasValue))
            {
                double? discoveredStop;
                double? discoveredTarget;
                DiscoverLiveProtectionPrices(account, instrument, position, out discoveredStop, out discoveredTarget);
                if (!stopPrice.HasValue && discoveredStop.HasValue)
                    stopPrice = discoveredStop;
                if (!targetPrice.HasValue && discoveredTarget.HasValue)
                    targetPrice = discoveredTarget;
            }

            double? lastPrice;
            double? bidPrice;
            double? askPrice;
            ReadMarketSnapshot(instrument, out lastPrice, out bidPrice, out askPrice);

            double? positionUnrealizedPnl = hasPosition
                ? (double?)position.GetUnrealizedProfitLoss(PerformanceUnit.Currency)
                : 0.0;
            double? accountRealizedPnl = ReadAccountItem(account, AccountItem.RealizedProfitLoss);
            double? accountUnrealizedPnl = ReadAccountItem(account, AccountItem.UnrealizedProfitLoss);
            double? accountTotalPnl = null;
            if (accountRealizedPnl.HasValue && accountUnrealizedPnl.HasValue)
                accountTotalPnl = accountRealizedPnl.Value + accountUnrealizedPnl.Value;

            return new BridgeResponse
            {
                Ok = true,
                Code = "state",
                Message = "state_ok",
                TimestampUtc = DateTime.UtcNow.ToString("o"),
                Account = account.Name,
                AccountCurrency = account.Denomination.ToString(),
                Symbol = instrument != null ? instrument.FullName : (symbol ?? string.Empty),
                MarketPosition = marketPosition,
                Quantity = hasPosition ? Math.Abs(position.Quantity) : 0,
                AveragePrice = hasPosition ? (double?)position.AveragePrice : null,
                StopPrice = stopPrice,
                TargetPrice = targetPrice,
                LastPrice = lastPrice,
                BidPrice = bidPrice,
                AskPrice = askPrice,
                PositionUnrealizedPnl = positionUnrealizedPnl,
                AccountRealizedPnl = accountRealizedPnl,
                AccountUnrealizedPnl = accountUnrealizedPnl,
                AccountTotalPnl = accountTotalPnl,
                HasPosition = hasPosition,
            };
        }

        private BridgeResponse ExecuteCommand(BridgeCommand command)
        {
            if (string.IsNullOrWhiteSpace(command.Symbol))
                return BridgeResponse.Error("bad_request", "symbol is required.");
            if (string.IsNullOrWhiteSpace(command.Action))
                return BridgeResponse.Error("bad_request", "action is required.");

            string accountName = !string.IsNullOrWhiteSpace(command.Account) ? command.Account : DefaultAccountName;
            Account account = ResolveAccount(accountName);
            if (account == null)
                return BridgeResponse.Error("account_not_found", "Account not found: " + accountName);

            SubscribeAccountEvents(account);

            Instrument instrument = Instrument.GetInstrument(command.Symbol);
            if (instrument == null)
                return BridgeResponse.Error("instrument_not_found", "Instrument not found: " + command.Symbol + ". Use exact NinjaTrader instrument (e.g. NQ 06-26).");

            string action = command.Action.Trim().ToUpperInvariant();
            switch (action)
            {
                case "ENTER_LONG":
                    return SubmitEntry(account, instrument, OrderAction.Buy, command);
                case "ENTER_SHORT":
                    return SubmitEntry(account, instrument, OrderAction.SellShort, command);
                case "ADD":
                    return SubmitAdd(account, instrument, command);
                case "TRIM":
                    return SubmitTrim(account, instrument, command);
                case "EXIT_ALL":
                    return SubmitExitAll(account, instrument);
                case "MOVE_STOP":
                    return SubmitMoveStop(account, instrument, command.StopPrice);
                case "MOVE_TO_BREAKEVEN":
                    var position = FindPosition(account, instrument);
                    if (position == null || position.MarketPosition == MarketPosition.Flat)
                        return BridgeResponse.Error("no_position", "Cannot move stop to breakeven while flat.");
                    return SubmitMoveStop(account, instrument, position.AveragePrice);
                default:
                    return BridgeResponse.Error("unsupported_action", "Unsupported action: " + command.Action);
            }
        }

        private BridgeResponse SubmitEntry(Account account, Instrument instrument, OrderAction action, BridgeCommand command)
        {
            int quantity = Math.Max(1, command.DefaultContractSize > 0 ? command.DefaultContractSize : 1);
            TimeInForce tif = ParseTimeInForce(command.TimeInForce);
            string signalName = BuildEntrySignalName(command.IntentId);
            double? effectiveStopPrice = command.StopPrice;
            double? effectiveTargetPrice = command.TargetPrice;

            FillMissingEntryProtectionPrices(instrument, action, command, ref effectiveStopPrice, ref effectiveTargetPrice);
            if (HasMeaningfulChange(command.StopPrice, effectiveStopPrice) || HasMeaningfulChange(command.TargetPrice, effectiveTargetPrice))
            {
                Print("[StreamCopierBridge] Filled missing entry protections for "
                    + instrument.FullName
                    + " reqStop=" + FormatNullablePrice(command.StopPrice)
                    + " reqTarget=" + FormatNullablePrice(command.TargetPrice)
                    + " -> stop=" + FormatNullablePrice(effectiveStopPrice)
                    + " target=" + FormatNullablePrice(effectiveTargetPrice));
            }

            bool hasProtection = effectiveStopPrice.HasValue || effectiveTargetPrice.HasValue;
            string protectionKey = hasProtection ? BuildProtectionKey(account, instrument) : null;

            try
            {
                if (hasProtection)
                {
                    lock (sync)
                    {
                        // Register before submit to avoid race on fast fills.
                        pendingProtections[protectionKey] = new PendingProtection
                        {
                            AccountName = account.Name,
                            InstrumentFullName = instrument.FullName,
                            StopPrice = effectiveStopPrice,
                            TargetPrice = effectiveTargetPrice,
                            TimeInForce = tif,
                        };
                    }
                }

                var order = account.CreateOrder(
                    instrument,
                    action,
                    OrderType.Market,
                    tif,
                    quantity,
                    0,
                    0,
                    string.Empty,
                    signalName,
                    null);

                account.Submit(new[] { order });

                return BridgeResponse.OkResult("entry_submitted", "Submitted " + action + " " + quantity + " " + instrument.FullName + ".");
            }
            catch (Exception ex)
            {
                if (hasProtection && !string.IsNullOrEmpty(protectionKey))
                {
                    lock (sync)
                    {
                        if (pendingProtections.ContainsKey(protectionKey))
                            pendingProtections.Remove(protectionKey);
                    }
                }
                return BridgeResponse.Error("submit_failed", ex.Message);
            }
        }

        private BridgeResponse SubmitAdd(Account account, Instrument instrument, BridgeCommand command)
        {
            if (!string.IsNullOrWhiteSpace(command.Side))
            {
                OrderAction side = ParseSideToEntryAction(command.Side);
                return SubmitEntry(account, instrument, side, command);
            }

            var position = FindPosition(account, instrument);
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return BridgeResponse.Error("no_position", "ADD requires side or existing position.");

            return SubmitEntry(
                account,
                instrument,
                position.MarketPosition == MarketPosition.Long ? OrderAction.Buy : OrderAction.SellShort,
                command);
        }

        private BridgeResponse SubmitTrim(Account account, Instrument instrument, BridgeCommand command)
        {
            var position = FindPosition(account, instrument);
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return BridgeResponse.Error("no_position", "Cannot TRIM while flat.");

            int currentQty = Math.Abs(position.Quantity);
            int trimQty = ResolveTrimQuantity(command.QuantityHint, currentQty);
            if (trimQty <= 0)
                return BridgeResponse.Error("bad_quantity", "Trim quantity resolved to zero.");

            var action = position.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;
            TimeInForce tif = ParseTimeInForce(command.TimeInForce);

            try
            {
                var order = account.CreateOrder(
                    instrument,
                    action,
                    OrderType.Market,
                    tif,
                    trimQty,
                    0,
                    0,
                    string.Empty,
                    "SC_TRIM_" + Guid.NewGuid().ToString("N").Substring(0, 10),
                    null);

                account.Submit(new[] { order });
                return BridgeResponse.OkResult("trim_submitted", "Submitted trim " + trimQty + " " + instrument.FullName + ".");
            }
            catch (Exception ex)
            {
                return BridgeResponse.Error("submit_failed", ex.Message);
            }
        }

        private BridgeResponse SubmitExitAll(Account account, Instrument instrument)
        {
            try
            {
                account.Flatten(new[] { instrument });
                CancelTrackedProtection(account, instrument);
                return BridgeResponse.OkResult("flatten_submitted", "Flatten submitted for " + instrument.FullName + ".");
            }
            catch (Exception ex)
            {
                return BridgeResponse.Error("flatten_failed", ex.Message);
            }
        }

        private BridgeResponse SubmitMoveStop(Account account, Instrument instrument, double? stopPrice)
        {
            if (!stopPrice.HasValue || stopPrice.Value <= 0)
                return BridgeResponse.Error("bad_request", "stop_price is required for MOVE_STOP.");

            var position = FindPosition(account, instrument);
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return BridgeResponse.Error("no_position", "Cannot move stop while flat.");

            try
            {
                UpsertProtectionOrders(account, instrument, position, stopPrice, null, TimeInForce.Day);
                return BridgeResponse.OkResult("stop_updated", "Stop moved to " + stopPrice.Value.ToString("0.00") + ".");
            }
            catch (Exception ex)
            {
                return BridgeResponse.Error("stop_update_failed", ex.Message);
            }
        }

        private void SubscribeAccountEvents(Account account)
        {
            lock (sync)
            {
                if (subscribedAccounts.Contains(account.Name))
                    return;

                account.OrderUpdate += OnOrderUpdate;
                account.ExecutionUpdate += OnExecutionUpdate;
                account.PositionUpdate += OnPositionUpdate;
                subscribedAccounts.Add(account.Name);
            }
        }

        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            if (e == null || e.Order == null || e.Order.Instrument == null)
                return;

            var account = sender as Account;
            if (account == null)
                return;

            // If an entry order is rejected/cancelled, clear pending protections so they
            // cannot leak into a later unrelated position on the same instrument.
            if ((e.OrderState == OrderState.Rejected || e.OrderState == OrderState.Cancelled)
                && IsEntryOrderAction(e.Order.OrderAction))
            {
                string key = BuildProtectionKey(account, e.Order.Instrument);
                lock (sync)
                {
                    if (pendingProtections.ContainsKey(key))
                        pendingProtections.Remove(key);
                }
            }

            // Protection attachment is handled in OnExecutionUpdate/OnPositionUpdate so we
            // don't rely on any particular Order.Name/SignalName behavior.
        }

        private void OnExecutionUpdate(object sender, ExecutionEventArgs e)
        {
            if (e == null || e.Execution == null || e.Execution.Instrument == null)
                return;

            var account = sender as Account;
            if (account == null)
                return;

            string key = BuildProtectionKey(account, e.Execution.Instrument);
            PendingProtection pending = null;
            lock (sync)
            {
                pendingProtections.TryGetValue(key, out pending);
                if (pending != null)
                    pendingProtections.Remove(key);
            }

            if (pending == null)
                return;

            try
            {
                var position = FindPosition(account, e.Execution.Instrument);
                if (position == null || position.MarketPosition == MarketPosition.Flat || position.Quantity == 0)
                {
                    // Execution arrived before position snapshot reflects size; retry on next event.
                    lock (sync)
                    {
                        pendingProtections[key] = pending;
                    }
                    return;
                }

                UpsertProtectionOrders(account, e.Execution.Instrument, position, pending.StopPrice, pending.TargetPrice, pending.TimeInForce);
            }
            catch (Exception ex)
            {
                Print("[StreamCopierBridge] Failed to place protection orders on execution: " + ex.Message);
                lock (sync)
                {
                    pendingProtections[key] = pending;
                }
            }
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            if (e == null || e.Position == null || e.Position.Instrument == null)
                return;

            var account = sender as Account;
            if (account == null)
                return;

            string key = BuildProtectionKey(account, e.Position.Instrument);
            PendingProtection pending = null;
            ProtectionState state = null;

            lock (sync)
            {
                // Consume pending protection only once, when position becomes non-flat.
                if (e.MarketPosition != MarketPosition.Flat && e.Quantity != 0)
                {
                    pendingProtections.TryGetValue(key, out pending);
                    if (pending != null)
                        pendingProtections.Remove(key);
                }

                protectionBySymbol.TryGetValue(key, out state);
            }

            if (pending != null)
            {
                try
                {
                    UpsertProtectionOrders(account, e.Position.Instrument, e.Position, pending.StopPrice, pending.TargetPrice, pending.TimeInForce);
                }
                catch (Exception ex)
                {
                    Print("[StreamCopierBridge] Failed to place protection orders: " + ex.Message);
                }
                return;
            }

            if (state == null)
                return;

            if (e.MarketPosition == MarketPosition.Flat || e.Quantity == 0)
            {
                CancelTrackedProtection(account, e.Position.Instrument);
                return;
            }

            int newQty = Math.Abs(e.Quantity);
            try
            {
                if (state.StopOrder != null && !IsTerminal(state.StopOrder))
                {
                    state.StopOrder.QuantityChanged = newQty;
                    account.Change(new[] { state.StopOrder });
                }
                if (state.TargetOrder != null && !IsTerminal(state.TargetOrder))
                {
                    state.TargetOrder.QuantityChanged = newQty;
                    account.Change(new[] { state.TargetOrder });
                }
            }
            catch (Exception ex)
            {
                Print("[StreamCopierBridge] Failed to sync protection quantity: " + ex.Message);
            }
        }

        private void UpsertProtectionOrders(Account account, Instrument instrument, Position position, double? stopPrice, double? targetPrice, TimeInForce tif)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return;

            double? requestedStop = stopPrice;
            double? requestedTarget = targetPrice;
            double? normalizedStop = stopPrice;
            double? normalizedTarget = targetPrice;
            NormalizeProtectionPrices(instrument, position, ref normalizedStop, ref normalizedTarget);
            stopPrice = normalizedStop;
            targetPrice = normalizedTarget;

            if (HasMeaningfulChange(requestedStop, stopPrice) || HasMeaningfulChange(requestedTarget, targetPrice))
            {
                Print("[StreamCopierBridge] Normalized protection levels for "
                    + instrument.FullName
                    + " reqStop=" + FormatNullablePrice(requestedStop)
                    + " reqTarget=" + FormatNullablePrice(requestedTarget)
                    + " -> stop=" + FormatNullablePrice(stopPrice)
                    + " target=" + FormatNullablePrice(targetPrice));
            }

            string key = BuildProtectionKey(account, instrument);
            ProtectionState existing = null;
            lock (sync)
            {
                protectionBySymbol.TryGetValue(key, out existing);
            }

            if (existing != null)
                CancelTrackedProtection(account, instrument);

            int qty = Math.Abs(position.Quantity);
            if (qty <= 0)
                return;

            var orders = new List<Order>();
            string oco = "SC_OCO_" + Guid.NewGuid().ToString("N").Substring(0, 16);
            OrderAction protectiveAction = position.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;

            if (stopPrice.HasValue && stopPrice.Value > 0)
            {
                var stopOrder = account.CreateOrder(
                    instrument,
                    protectiveAction,
                    OrderType.StopMarket,
                    tif,
                    qty,
                    0,
                    stopPrice.Value,
                    oco,
                    "SC_STOP_" + Guid.NewGuid().ToString("N").Substring(0, 8),
                    null);
                orders.Add(stopOrder);
            }

            if (targetPrice.HasValue && targetPrice.Value > 0)
            {
                var targetOrder = account.CreateOrder(
                    instrument,
                    protectiveAction,
                    OrderType.Limit,
                    tif,
                    qty,
                    targetPrice.Value,
                    0,
                    oco,
                    "SC_TARGET_" + Guid.NewGuid().ToString("N").Substring(0, 8),
                    null);
                orders.Add(targetOrder);
            }

            if (orders.Count == 0)
                return;

            account.Submit(orders.ToArray());

            lock (sync)
            {
                protectionBySymbol[key] = new ProtectionState
                {
                    StopOrder = orders.Find(o => o.OrderType == OrderType.StopMarket),
                    TargetOrder = orders.Find(o => o.OrderType == OrderType.Limit),
                    StopPrice = stopPrice,
                    TargetPrice = targetPrice,
                };
            }
        }

        private void CancelTrackedProtection(Account account, Instrument instrument)
        {
            string key = BuildProtectionKey(account, instrument);
            ProtectionState state = null;

            lock (sync)
            {
                if (protectionBySymbol.TryGetValue(key, out state))
                    protectionBySymbol.Remove(key);
            }

            if (state == null)
                return;

            try
            {
                var toCancel = new List<Order>();
                if (state.StopOrder != null && !IsTerminal(state.StopOrder))
                    toCancel.Add(state.StopOrder);
                if (state.TargetOrder != null && !IsTerminal(state.TargetOrder))
                    toCancel.Add(state.TargetOrder);

                if (toCancel.Count > 0)
                    account.Cancel(toCancel.ToArray());
            }
            catch (Exception ex)
            {
                Print("[StreamCopierBridge] Failed to cancel tracked protection orders: " + ex.Message);
            }
        }

        private Position FindPosition(Account account, Instrument instrument)
        {
            if (account == null || instrument == null)
                return null;

            lock (account.Positions)
            {
                foreach (Position p in account.Positions)
                {
                    if (p == null || p.Instrument == null)
                        continue;
                    if (string.Equals(p.Instrument.FullName, instrument.FullName, StringComparison.OrdinalIgnoreCase))
                        return p;
                }
            }
            return null;
        }

        private static Instrument FindFirstOpenInstrument(Account account)
        {
            if (account == null)
                return null;

            lock (account.Positions)
            {
                foreach (Position p in account.Positions)
                {
                    if (p == null || p.Instrument == null)
                        continue;
                    if (p.MarketPosition != MarketPosition.Flat && Math.Abs(p.Quantity) > 0)
                        return p.Instrument;
                }
            }

            return null;
        }

        private static void ReadMarketSnapshot(Instrument instrument, out double? lastPrice, out double? bidPrice, out double? askPrice)
        {
            lastPrice = null;
            bidPrice = null;
            askPrice = null;

            if (instrument == null)
                return;

            try
            {
                if (instrument.MarketData == null)
                    return;

                if (instrument.MarketData.Last != null)
                    lastPrice = instrument.MarketData.Last.Price;
                if (instrument.MarketData.Bid != null)
                    bidPrice = instrument.MarketData.Bid.Price;
                if (instrument.MarketData.Ask != null)
                    askPrice = instrument.MarketData.Ask.Price;
            }
            catch
            {
                // Best effort market snapshot endpoint.
            }
        }

        private static void FillMissingEntryProtectionPrices(
            Instrument instrument,
            OrderAction entryAction,
            BridgeCommand command,
            ref double? stopPrice,
            ref double? targetPrice)
        {
            if (entryAction != OrderAction.Buy && entryAction != OrderAction.SellShort)
                return;
            if (stopPrice.HasValue && targetPrice.HasValue)
                return;

            double? referencePrice = command != null ? command.EntryPrice : null;
            if (!referencePrice.HasValue || referencePrice.Value <= 0)
            {
                double? lastPrice;
                double? bidPrice;
                double? askPrice;
                ReadMarketSnapshot(instrument, out lastPrice, out bidPrice, out askPrice);
                referencePrice = ResolveEntryReferencePrice(lastPrice, bidPrice, askPrice, entryAction);
            }

            if (!referencePrice.HasValue || referencePrice.Value <= 0)
                return;

            double tickSize = ResolveTickSize(instrument);
            double stopOffset = Math.Max(FallbackWideStopPoints, tickSize * 40.0);
            double targetOffset = Math.Max(FallbackWideTargetPoints, stopOffset);

            if (entryAction == OrderAction.Buy)
            {
                if (!stopPrice.HasValue || stopPrice.Value <= 0)
                    stopPrice = RoundToTick(instrument, referencePrice.Value - stopOffset, false);
                if (!targetPrice.HasValue || targetPrice.Value <= 0)
                    targetPrice = RoundToTick(instrument, referencePrice.Value + targetOffset, true);
            }
            else
            {
                if (!stopPrice.HasValue || stopPrice.Value <= 0)
                    stopPrice = RoundToTick(instrument, referencePrice.Value + stopOffset, true);
                if (!targetPrice.HasValue || targetPrice.Value <= 0)
                    targetPrice = RoundToTick(instrument, referencePrice.Value - targetOffset, false);
            }
        }

        private static double? ResolveEntryReferencePrice(double? lastPrice, double? bidPrice, double? askPrice, OrderAction entryAction)
        {
            if (entryAction == OrderAction.Buy)
            {
                if (askPrice.HasValue && askPrice.Value > 0)
                    return askPrice.Value;
                if (lastPrice.HasValue && lastPrice.Value > 0)
                    return lastPrice.Value;
                if (bidPrice.HasValue && bidPrice.Value > 0)
                    return bidPrice.Value;
                return null;
            }

            if (entryAction == OrderAction.SellShort)
            {
                if (bidPrice.HasValue && bidPrice.Value > 0)
                    return bidPrice.Value;
                if (lastPrice.HasValue && lastPrice.Value > 0)
                    return lastPrice.Value;
                if (askPrice.HasValue && askPrice.Value > 0)
                    return askPrice.Value;
                return null;
            }

            return null;
        }

        private static void DiscoverLiveProtectionPrices(
            Account account,
            Instrument instrument,
            Position position,
            out double? stopPrice,
            out double? targetPrice)
        {
            stopPrice = null;
            targetPrice = null;

            if (account == null || instrument == null || position == null)
                return;
            if (position.MarketPosition == MarketPosition.Flat || position.Quantity == 0)
                return;

            OrderAction protectiveAction = position.MarketPosition == MarketPosition.Long
                ? OrderAction.Sell
                : OrderAction.BuyToCover;

            var preferredStops = new List<Order>();
            var preferredTargets = new List<Order>();
            var fallbackStops = new List<Order>();
            var fallbackTargets = new List<Order>();

            lock (account.Orders)
            {
                foreach (Order order in account.Orders)
                {
                    if (order == null || order.Instrument == null)
                        continue;
                    if (!string.Equals(order.Instrument.FullName, instrument.FullName, StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (order.OrderAction != protectiveAction)
                        continue;
                    if (IsTerminal(order))
                        continue;

                    string name = order.Name ?? string.Empty;
                    bool isBridgeStop = name.StartsWith("SC_STOP_", StringComparison.OrdinalIgnoreCase);
                    bool isBridgeTarget = name.StartsWith("SC_TARGET_", StringComparison.OrdinalIgnoreCase);
                    bool isStopType = order.OrderType == OrderType.StopMarket || order.OrderType == OrderType.StopLimit;
                    bool isTargetType = order.OrderType == OrderType.Limit;

                    if (isStopType)
                    {
                        if (isBridgeStop)
                            preferredStops.Add(order);
                        else
                            fallbackStops.Add(order);
                    }
                    else if (isTargetType)
                    {
                        if (isBridgeTarget)
                            preferredTargets.Add(order);
                        else
                            fallbackTargets.Add(order);
                    }
                }
            }

            Order stopOrder = preferredStops.Count > 0 ? preferredStops[0] : (fallbackStops.Count > 0 ? fallbackStops[0] : null);
            Order targetOrder = preferredTargets.Count > 0 ? preferredTargets[0] : (fallbackTargets.Count > 0 ? fallbackTargets[0] : null);

            if (stopOrder != null)
                stopPrice = stopOrder.StopPrice;
            if (targetOrder != null)
                targetPrice = targetOrder.LimitPrice;
        }

        private static void NormalizeProtectionPrices(Instrument instrument, Position position, ref double? stopPrice, ref double? targetPrice)
        {
            if (instrument == null || position == null || position.MarketPosition == MarketPosition.Flat)
                return;

            double? lastPrice;
            double? bidPrice;
            double? askPrice;
            ReadMarketSnapshot(instrument, out lastPrice, out bidPrice, out askPrice);

            double? market = ResolveReferenceMarket(lastPrice, bidPrice, askPrice, position.MarketPosition);
            double tickSize = ResolveTickSize(instrument);
            double avg = position.AveragePrice;
            double defaultDistance = tickSize * 40.0; // keep brackets meaningfully wide when fallback is needed
            bool isLong = position.MarketPosition == MarketPosition.Long;
            if (market.HasValue)
            {
                double minGap = tickSize * 2.0;
                double mkt = market.Value;

                if (stopPrice.HasValue)
                {
                    double stopDistance = Math.Max(Math.Abs(stopPrice.Value - avg), defaultDistance);
                    if (isLong)
                    {
                        if (stopPrice.Value >= mkt - minGap)
                            stopPrice = RoundToTick(instrument, mkt - stopDistance, false);
                        else
                            stopPrice = RoundToTick(instrument, stopPrice.Value, false);
                    }
                    else
                    {
                        if (stopPrice.Value <= mkt + minGap)
                            stopPrice = RoundToTick(instrument, mkt + stopDistance, true);
                        else
                            stopPrice = RoundToTick(instrument, stopPrice.Value, true);
                    }
                }

                if (targetPrice.HasValue)
                {
                    double targetDistance = Math.Max(Math.Abs(targetPrice.Value - avg), defaultDistance);
                    if (isLong)
                    {
                        if (targetPrice.Value <= mkt + minGap)
                            targetPrice = RoundToTick(instrument, mkt + targetDistance, true);
                        else
                            targetPrice = RoundToTick(instrument, targetPrice.Value, true);
                    }
                    else
                    {
                        if (targetPrice.Value >= mkt - minGap)
                            targetPrice = RoundToTick(instrument, mkt - targetDistance, false);
                        else
                            targetPrice = RoundToTick(instrument, targetPrice.Value, false);
                    }
                }
            }

            EnsureAllProtectionPrices(instrument, position, ref stopPrice, ref targetPrice);
        }

        private static void EnsureAllProtectionPrices(Instrument instrument, Position position, ref double? stopPrice, ref double? targetPrice)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return;
            if (stopPrice.HasValue && targetPrice.HasValue)
                return;

            double tickSize = ResolveTickSize(instrument);
            double stopOffset = Math.Max(FallbackWideStopPoints, tickSize * 40.0);
            double targetOffset = Math.Max(FallbackWideTargetPoints, stopOffset);
            double avg = position.AveragePrice;
            bool isLong = position.MarketPosition == MarketPosition.Long;

            if (!stopPrice.HasValue || stopPrice.Value <= 0)
            {
                stopPrice = isLong
                    ? RoundToTick(instrument, avg - stopOffset, false)
                    : RoundToTick(instrument, avg + stopOffset, true);
            }

            if (!targetPrice.HasValue || targetPrice.Value <= 0)
            {
                targetPrice = isLong
                    ? RoundToTick(instrument, avg + targetOffset, true)
                    : RoundToTick(instrument, avg - targetOffset, false);
            }
        }

        private static double? ResolveReferenceMarket(double? lastPrice, double? bidPrice, double? askPrice, MarketPosition positionSide)
        {
            var candidates = new List<double>();
            if (lastPrice.HasValue && !double.IsNaN(lastPrice.Value) && !double.IsInfinity(lastPrice.Value))
                candidates.Add(lastPrice.Value);
            if (bidPrice.HasValue && !double.IsNaN(bidPrice.Value) && !double.IsInfinity(bidPrice.Value))
                candidates.Add(bidPrice.Value);
            if (askPrice.HasValue && !double.IsNaN(askPrice.Value) && !double.IsInfinity(askPrice.Value))
                candidates.Add(askPrice.Value);

            if (candidates.Count == 0)
                return null;

            // Conservative market reference:
            // - Long protections use the lower boundary
            // - Short protections use the upper boundary
            if (positionSide == MarketPosition.Long)
                return Min(candidates);
            return Max(candidates);
        }

        private static double ResolveTickSize(Instrument instrument)
        {
            if (instrument != null && instrument.MasterInstrument != null && instrument.MasterInstrument.TickSize > 0)
                return instrument.MasterInstrument.TickSize;
            return 0.25;
        }

        private static double RoundToTick(Instrument instrument, double price, bool roundUp)
        {
            double tick = ResolveTickSize(instrument);
            if (tick <= 0)
                return price;
            double steps = price / tick;
            double roundedSteps = roundUp ? Math.Ceiling(steps) : Math.Floor(steps);
            return roundedSteps * tick;
        }

        private static double Min(List<double> values)
        {
            if (values == null || values.Count == 0)
                return 0;
            double min = values[0];
            for (int i = 1; i < values.Count; i++)
            {
                if (values[i] < min)
                    min = values[i];
            }
            return min;
        }

        private static double Max(List<double> values)
        {
            if (values == null || values.Count == 0)
                return 0;
            double max = values[0];
            for (int i = 1; i < values.Count; i++)
            {
                if (values[i] > max)
                    max = values[i];
            }
            return max;
        }

        private static bool HasMeaningfulChange(double? before, double? after)
        {
            if (!before.HasValue && !after.HasValue)
                return false;
            if (before.HasValue != after.HasValue)
                return true;
            return Math.Abs(before.Value - after.Value) > 0.0000001;
        }

        private static string FormatNullablePrice(double? value)
        {
            if (!value.HasValue)
                return "null";
            return value.Value.ToString("0.##########", CultureInfo.InvariantCulture);
        }

        private static double? ReadAccountItem(Account account, AccountItem itemType)
        {
            if (account == null)
                return null;

            try
            {
                double value = account.Get(itemType, account.Denomination);
                if (double.IsNaN(value) || double.IsInfinity(value))
                    return null;
                return value;
            }
            catch
            {
                return null;
            }
        }

        private static int ResolveTrimQuantity(string quantityHint, int currentQty)
        {
            if (currentQty <= 0)
                return 0;
            if (string.IsNullOrWhiteSpace(quantityHint))
                return currentQty > 1 ? 1 : currentQty;

            string hint = quantityHint.Trim().ToLowerInvariant();
            if (hint == "all")
                return currentQty;
            if (hint == "half")
                return Math.Max(1, currentQty / 2);
            if (hint == "most")
                return Math.Max(1, currentQty - 1);
            if (hint == "one")
                return 1;
            if (hint == "two")
                return Math.Min(2, currentQty);
            if (hint == "three")
                return Math.Min(3, currentQty);
            int parsed;
            if (int.TryParse(hint, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed))
            {
                if (parsed <= 0)
                    return 0;
                return Math.Min(parsed, currentQty);
            }
            return currentQty > 1 ? 1 : currentQty;
        }

        private static OrderAction ParseSideToEntryAction(string side)
        {
            if (string.IsNullOrWhiteSpace(side))
                return OrderAction.Buy; // fallback; ADD without side will infer from position before this is called

            string normalized = side.Trim().ToUpperInvariant();
            if (normalized == "LONG")
                return OrderAction.Buy;
            if (normalized == "SHORT")
                return OrderAction.SellShort;
            return OrderAction.Buy;
        }

        private static TimeInForce ParseTimeInForce(string tif)
        {
            if (string.IsNullOrWhiteSpace(tif))
                return TimeInForce.Day;

            string normalized = tif.Trim().ToUpperInvariant();
            if (normalized == "GTC")
                return TimeInForce.Gtc;
            return TimeInForce.Day;
        }

        private static bool IsTerminal(Order order)
        {
            if (order == null)
                return true;
            return order.OrderState == OrderState.Filled
                || order.OrderState == OrderState.Cancelled
                || order.OrderState == OrderState.Rejected
                || order.OrderState == OrderState.Unknown;
        }

        private static bool IsEntryOrderAction(OrderAction action)
        {
            return action == OrderAction.Buy || action == OrderAction.SellShort;
        }

        private static string BuildEntrySignalName(string intentId)
        {
            string suffix = !string.IsNullOrWhiteSpace(intentId)
                ? intentId.Replace("-", string.Empty)
                : Guid.NewGuid().ToString("N");
            if (suffix.Length > 18)
                suffix = suffix.Substring(0, 18);
            return "SC_ENTRY_" + suffix;
        }

        private static string BuildProtectionKey(Account account, Instrument instrument)
        {
            return account.Name + "|" + instrument.FullName;
        }

        private static string EnsureTrailingSlash(string prefix)
        {
            if (string.IsNullOrWhiteSpace(prefix))
                return "http://127.0.0.1:18080/";
            return prefix.EndsWith("/") ? prefix : prefix + "/";
        }

        private static Account ResolveAccount(string accountName)
        {
            lock (Account.All)
            {
                if (!string.IsNullOrWhiteSpace(accountName))
                {
                    foreach (Account account in Account.All)
                    {
                        if (string.Equals(account.Name, accountName, StringComparison.OrdinalIgnoreCase))
                            return account;
                    }
                }

                foreach (Account account in Account.All)
                    return account;
            }
            return null;
        }

        private bool IsAuthorized(HttpListenerRequest request)
        {
            if (string.IsNullOrWhiteSpace(RequiredBearerToken))
                return true;
            string auth = request.Headers["Authorization"] ?? string.Empty;
            string expected = "Bearer " + RequiredBearerToken;
            return string.Equals(auth, expected, StringComparison.Ordinal);
        }

        private BridgeResponse BuildHealthResponse()
        {
            var accounts = new List<string>();
            lock (Account.All)
            {
                foreach (Account account in Account.All)
                    accounts.Add(account.Name);
            }

            return new BridgeResponse
            {
                Ok = true,
                Code = "ok",
                Message = "bridge_up",
                TimestampUtc = DateTime.UtcNow.ToString("o"),
                Accounts = accounts,
            };
        }

        private static async Task WriteJsonAsync(HttpListenerContext context, int statusCode, BridgeResponse response)
        {
            string json = SerializeResponse(response);
            byte[] data = Encoding.UTF8.GetBytes(json);
            context.Response.StatusCode = statusCode;
            context.Response.ContentType = "application/json";
            context.Response.ContentEncoding = Encoding.UTF8;
            context.Response.ContentLength64 = data.Length;
            try
            {
                await context.Response.OutputStream.WriteAsync(data, 0, data.Length).ConfigureAwait(false);
            }
            finally
            {
                try { context.Response.OutputStream.Close(); } catch { }
                try { context.Response.Close(); } catch { }
            }
        }

        private static BridgeCommand DeserializeCommand(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return null;

            return new BridgeCommand
            {
                IntentId = ReadJsonString(json, "intent_id"),
                SessionId = ReadJsonString(json, "session_id"),
                Account = ReadJsonString(json, "account"),
                Symbol = ReadJsonString(json, "symbol"),
                Action = ReadJsonString(json, "action"),
                Side = ReadJsonString(json, "side"),
                QuantityHint = ReadJsonString(json, "quantity_hint"),
                DefaultContractSize = ReadJsonInt(json, "default_contract_size"),
                TimeInForce = ReadJsonString(json, "time_in_force"),
                EntryPrice = ReadJsonNullableDouble(json, "entry_price"),
                StopPrice = ReadJsonNullableDouble(json, "stop_price"),
                TargetPrice = ReadJsonNullableDouble(json, "target_price"),
            };
        }

        private static string SerializeResponse(BridgeResponse response)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append("\"ok\":").Append(response.Ok ? "true" : "false");
            sb.Append(",\"code\":\"").Append(JsonEscape(response.Code ?? string.Empty)).Append("\"");
            sb.Append(",\"message\":\"").Append(JsonEscape(response.Message ?? string.Empty)).Append("\"");
            sb.Append(",\"timestamp_utc\":\"").Append(JsonEscape(response.TimestampUtc ?? string.Empty)).Append("\"");

            if (response.Accounts != null && response.Accounts.Count > 0)
            {
                sb.Append(",\"accounts\":[");
                for (int i = 0; i < response.Accounts.Count; i++)
                {
                    if (i > 0)
                        sb.Append(",");
                    sb.Append("\"").Append(JsonEscape(response.Accounts[i] ?? string.Empty)).Append("\"");
                }
                sb.Append("]");
            }

            if (!string.IsNullOrWhiteSpace(response.Account))
                AppendJsonString(sb, "account", response.Account);
            if (!string.IsNullOrWhiteSpace(response.AccountCurrency))
                AppendJsonString(sb, "account_currency", response.AccountCurrency);
            if (!string.IsNullOrWhiteSpace(response.Symbol))
                AppendJsonString(sb, "symbol", response.Symbol);
            if (!string.IsNullOrWhiteSpace(response.MarketPosition))
                AppendJsonString(sb, "market_position", response.MarketPosition);
            AppendJsonInt(sb, "quantity", response.Quantity);
            AppendJsonNumber(sb, "average_price", response.AveragePrice);
            AppendJsonNumber(sb, "stop_price", response.StopPrice);
            AppendJsonNumber(sb, "target_price", response.TargetPrice);
            AppendJsonNumber(sb, "last_price", response.LastPrice);
            AppendJsonNumber(sb, "bid_price", response.BidPrice);
            AppendJsonNumber(sb, "ask_price", response.AskPrice);
            AppendJsonNumber(sb, "position_unrealized_pnl", response.PositionUnrealizedPnl);
            AppendJsonNumber(sb, "account_realized_pnl", response.AccountRealizedPnl);
            AppendJsonNumber(sb, "account_unrealized_pnl", response.AccountUnrealizedPnl);
            AppendJsonNumber(sb, "account_total_pnl", response.AccountTotalPnl);
            AppendJsonBool(sb, "has_position", response.HasPosition);

            sb.Append("}");
            return sb.ToString();
        }

        private static void AppendJsonString(StringBuilder sb, string name, string value)
        {
            if (string.IsNullOrEmpty(name) || value == null)
                return;
            sb.Append(",\"").Append(JsonEscape(name)).Append("\":\"").Append(JsonEscape(value)).Append("\"");
        }

        private static void AppendJsonNumber(StringBuilder sb, string name, double? value)
        {
            if (string.IsNullOrEmpty(name) || !value.HasValue)
                return;
            sb.Append(",\"").Append(JsonEscape(name)).Append("\":").Append(value.Value.ToString("0.##########", CultureInfo.InvariantCulture));
        }

        private static void AppendJsonInt(StringBuilder sb, string name, int? value)
        {
            if (string.IsNullOrEmpty(name) || !value.HasValue)
                return;
            sb.Append(",\"").Append(JsonEscape(name)).Append("\":").Append(value.Value.ToString(CultureInfo.InvariantCulture));
        }

        private static void AppendJsonBool(StringBuilder sb, string name, bool? value)
        {
            if (string.IsNullOrEmpty(name) || !value.HasValue)
                return;
            sb.Append(",\"").Append(JsonEscape(name)).Append("\":").Append(value.Value ? "true" : "false");
        }

        private static string ReadJsonString(string json, string key)
        {
            var match = Regex.Match(
                json,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"(?<v>(?:\\\\.|[^\"])*)\"",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!match.Success)
                return null;
            return JsonUnescape(match.Groups["v"].Value);
        }

        private static int ReadJsonInt(string json, string key)
        {
            var match = Regex.Match(
                json,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*(?<v>-?\\d+)",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!match.Success)
                return 0;
            int value;
            if (int.TryParse(match.Groups["v"].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
                return value;
            return 0;
        }

        private static double? ReadJsonNullableDouble(string json, string key)
        {
            var nullMatch = Regex.Match(
                json,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*null",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (nullMatch.Success)
                return null;

            var match = Regex.Match(
                json,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*(?<v>-?\\d+(?:\\.\\d+)?)",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!match.Success)
                return null;
            double value;
            if (double.TryParse(match.Groups["v"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
                return value;
            return null;
        }

        private static string JsonEscape(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;
            return value
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n")
                .Replace("\t", "\\t");
        }

        private static string JsonUnescape(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;
            return value
                .Replace("\\\"", "\"")
                .Replace("\\\\", "\\")
                .Replace("\\r", "\r")
                .Replace("\\n", "\n")
                .Replace("\\t", "\t");
        }

        private sealed class PendingProtection
        {
            public string AccountName { get; set; }
            public string InstrumentFullName { get; set; }
            public double? StopPrice { get; set; }
            public double? TargetPrice { get; set; }
            public TimeInForce TimeInForce { get; set; }
        }

        private sealed class ProtectionState
        {
            public Order StopOrder { get; set; }
            public Order TargetOrder { get; set; }
            public double? StopPrice { get; set; }
            public double? TargetPrice { get; set; }
        }

        private sealed class BridgeCommand
        {
            public string IntentId { get; set; }

            public string SessionId { get; set; }

            public string Account { get; set; }

            public string Symbol { get; set; }

            public string Action { get; set; }

            public string Side { get; set; }

            public string QuantityHint { get; set; }

            public int DefaultContractSize { get; set; }

            public string TimeInForce { get; set; }

            public double? EntryPrice { get; set; }

            public double? StopPrice { get; set; }

            public double? TargetPrice { get; set; }
        }

        private sealed class BridgeResponse
        {
            public bool Ok { get; set; }

            public string Code { get; set; }

            public string Message { get; set; }

            public string TimestampUtc { get; set; }

            public List<string> Accounts { get; set; }

            public string Account { get; set; }

            public string AccountCurrency { get; set; }

            public string Symbol { get; set; }

            public string MarketPosition { get; set; }

            public int? Quantity { get; set; }

            public double? AveragePrice { get; set; }

            public double? StopPrice { get; set; }

            public double? TargetPrice { get; set; }

            public double? LastPrice { get; set; }

            public double? BidPrice { get; set; }

            public double? AskPrice { get; set; }

            public double? PositionUnrealizedPnl { get; set; }

            public double? AccountRealizedPnl { get; set; }

            public double? AccountUnrealizedPnl { get; set; }

            public double? AccountTotalPnl { get; set; }

            public bool? HasPosition { get; set; }

            public static BridgeResponse OkResult(string code, string message)
            {
                return new BridgeResponse
                {
                    Ok = true,
                    Code = code,
                    Message = message,
                    TimestampUtc = DateTime.UtcNow.ToString("o"),
                };
            }

            public static BridgeResponse Error(string code, string message)
            {
                return new BridgeResponse
                {
                    Ok = false,
                    Code = code,
                    Message = message,
                    TimestampUtc = DateTime.UtcNow.ToString("o"),
                };
            }
        }
    }
}
