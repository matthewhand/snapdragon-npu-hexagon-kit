from hexagon_kit import provider_chain
from hexagon_kit.hw import CPU_PROVIDER, DML_PROVIDER, QNN_PROVIDER


def test_provider_chain_ends_at_cpu():
    chain = provider_chain()
    assert CPU_PROVIDER in chain
    assert chain[0] in {QNN_PROVIDER, DML_PROVIDER, CPU_PROVIDER}
    assert len(chain) == len(set(chain))


def test_provider_chain_cpu_ort_does_not_invent_qnn(monkeypatch):
    class _Probe:
        preferred_provider = CPU_PROVIDER
        providers = [CPU_PROVIDER]

    monkeypatch.setattr("hexagon_kit.session.probe_hardware", lambda: _Probe())
    chain = provider_chain()
    assert chain[0] == CPU_PROVIDER
    assert QNN_PROVIDER not in chain
    assert DML_PROVIDER not in chain


def test_provider_chain_keeps_qnn_ahead_of_directml(monkeypatch):
    class _Probe:
        preferred_provider = QNN_PROVIDER
        providers = [QNN_PROVIDER, DML_PROVIDER, CPU_PROVIDER]

    monkeypatch.setattr("hexagon_kit.session.probe_hardware", lambda: _Probe())
    chain = provider_chain()
    assert chain[0] == QNN_PROVIDER
    assert chain.index(DML_PROVIDER) < chain.index(CPU_PROVIDER)
