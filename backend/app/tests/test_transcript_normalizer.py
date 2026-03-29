from app.services.interpretation.transcript_normalizer import apply_trading_asr_corrections


def test_apply_trading_asr_corrections_normalizes_common_trading_confusions() -> None:
    corrected = apply_trading_asr_corrections(
        "Holding view up. Peace on short here. Runner break even. We're on the M N Q."
    )

    assert "vwap" in corrected
    assert "piece on short here" in corrected
    assert "runner breakeven" in corrected
    assert "mnq" in corrected
