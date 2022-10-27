import awkward as ak

## Jet pu ID not exist in Winter22Run3 sample
def jet_id(events, campaign):
    if campaign != "Winter22Run3":
        jetmask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) <= 2.5)
            & (events.Jet.jetId >= 5)
            & ((events.Jet.pt > 50) | (events.Jet.puId >= 7))
        )
    else:
        jetmask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) <= 2.5)
            & (events.Jet.jetId >= 5)
        )
    return jetmask


## FIXME: Electron cutbased Id & MVA ID not exist in Winter22Run3 sample
def ele_cuttightid(events, campaign):
    if campaign != "Winter22Run3":
        elemask = (
            (abs(events.Electron.eta) < 1.4442)
            | ((abs(events.Electron.eta) < 2.5) & (abs(events.Electron.eta) > 1.566))
        ) & (events.Electron.cutBased > 3)
    else:
        elemask = (abs(events.Electron.eta) < 1.4442) | (
            (abs(events.Electron.eta) < 2.5) & (abs(events.Electron.eta) > 1.566)
        )
    return elemask


def ele_mvatightid(events, campaign):
    if campaign != "Winter22Run3":
        elemask = (
            (abs(events.Electron.eta) < 1.4442)
            | ((abs(events.Electron.eta) < 2.5) & (abs(events.Electron.eta) > 1.566))
        ) & (events.Electron.mvaFall17V2Iso_WP80 > 0.5)
    else:
        elemask = (abs(events.Electron.eta) < 1.4442) | (
            (abs(events.Electron.eta) < 2.5) & (abs(events.Electron.eta) > 1.566)
        )
    return elemask


def softmu_mask(events, campaign):
    softmumask = (
        (events.Muon.pt < 25)
        & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.pfRelIso04_all > 0.2)
        & (events.Muon.jetIdx != -1)
    )

    return softmumask


def mu_idiso(events, campaign):
    mumask = (
        (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.pfRelIso04_all <= 0.15)
    )
    return mumask


def btag_mu_idiso(events, campaign):
    mumask = (
        (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.pfRelIso04_all < 0.12)
    )
    return mumask