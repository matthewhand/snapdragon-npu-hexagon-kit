from hexagon_kit import provider_chain


def test_provider_chain_ends_at_cpu():
    chain = provider_chain()
    assert "CPUExecutionProvider" in chain
    assert chain[0] in {
        "QNNExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    }
    assert len(chain) == len(set(chain))
