import coffea
from coffea import hist, processor
import numpy as np
#import awkward1 as ak
import awkward as ak
from coffea.analysis_tools import Weights


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self):        
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cutflow_axis   = hist.Cat("cut",   "Cut")
       
        # Events
        nel_axis   = hist.Bin("nel",   r"N electrons", [0,1,2,3,4,5,6,7,8,9,10])
        nmu_axis   = hist.Bin("nmu",   r"N muons",     [0,1,2,3,4,5,6,7,8,9,10])
        njet_axis  = hist.Bin("njet",  r"N jets",      [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_t_axis = hist.Bin("nbjet_t", r"N tight b-jets",    [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_m_axis = hist.Bin("nbjet_m", r"N medium b-jets",    [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_l_axis = hist.Bin("nbjet_l", r"N loose b-jets",    [0,1,2,3,4,5,6,7,8,9,10])

        # Electron
        el_pt_axis   = hist.Bin("pt",    r"Electron $p_{T}$ [GeV]", 100, 20, 400)
        el_eta_axis  = hist.Bin("eta",   r"$\eta$", 60, -3, 3)
        el_phi_axis  = hist.Bin("phi",   r"$\phi$", 60, -3, 3)
        lelpt_axis   = hist.Bin("lelpt", r"Leading electron $p_{T}$ [GeV]", 100, 20, 200)
        
        # Muons
        mu_pt_axis   = hist.Bin("pt",    r"Muon $p_{T}$ [GeV]", 100, 20, 400)
        mu_eta_axis  = hist.Bin("eta",   r"$\eta$", 60, -3, 3)
        mu_phi_axis  = hist.Bin("phi",   r"$\phi$", 60, -3, 3)
        lmupt_axis   = hist.Bin("lmupt", r"Leading muon $p_{T}$ [GeV]", 100, 20, 200)
        
        # Jet
        flav_axis = hist.Bin("flav", r"Genflavour",[0,4,5,6])
        jet_pt_axis   = hist.Bin("pt",   r"Jet $p_{T}$ [GeV]", 300, 0, 3000)
        jet_eta_axis  = hist.Bin("eta",  r"$\eta$", 60, -3, 3)
        jet_phi_axis  = hist.Bin("phi",  r"$\phi$", 60, -3, 3)
        jet_mass_axis = hist.Bin("mass", r"Jet $m$ [GeV]", 200, 0, 200)
        ljeta_axis  = hist.Bin("ljeta",  r"Leading jet $\eta$", 60, -3, 3)
        ljphi_axis  = hist.Bin("ljphi",  r"Leading jet $\phi$", 60, -3, 3)
        ljpt_axis     = hist.Bin("ljpt", r"Leading jet $p_{T}$ [GeV]", 3000, 0, 3000)
        ljpt_etacut_axis     = hist.Bin("ljpt_etacut", r"Leading jet $p_{T}$ without $p_{T}$ cut [GeV]", 3000, 0, 3000)
        jet_pt_etacut_axis     = hist.Bin("jet_pt_etacut", r"jet $p_{T}$ without $p_{T}$ cut [GeV]", 3000, 0, 3000)
        jet_pt_nocut_axis     = hist.Bin("jet_pt_nocut", r"jet $p_{T}$ without any cut [GeV]", 3000, 0, 3000)
        sljpt_axis     = hist.Bin("sljpt", r"Subleading jet $p_{T}$ [GeV]", 100, 20, 400)
        
        # PV 
        PV_npvsGood_axis = hist.Bin("PV_npvsGood", r"number of good reconstructed primary vertices", 100, 0, 100)
        Pileup_nTrueInt_axis = hist.Bin("Pileup_nTrueInt", r"nTrueInt", 100, 0, 100)
        
        #SV
        ntracks_axis   = hist.Bin("ntracks",   r"Number of trackss associated with SV", [0,1,2,3,4,5,6,7,8,9,10])
        SV_mass_axis   = hist.Bin("SV_mass",   r"invariant mass of the secondary vertex", 20, 0, 10)
        SV_dlen_axis = hist.Bin("SV_dlen",   r"decay length (cm)", 100, 0, 100)
        SV_dlenSig_axis = hist.Bin("SV_dlenSig",   r"decay length significance", 100, 0, 100)
        SV_deltaR_axis = hist.Bin("SV_deltaR",   r"dR from parent jet", 100, 0, 1.0)
        SV_pt_axis = hist.Bin("SV_pt",   r"SV $p_{T}$ [GeV]", 200, 0, 200)
        
        #DeepCSV, DeepJet descriminators
        lj_btagCMVA_axis = hist.Bin('lj_btagCMVA',r"Leading jet btagCMVA" , 50, 0, 1.0)
        lj_btagCSVV2_axis = hist.Bin('lj_btagCSVV2',r"Leading jet btagCSVV2" , 50, 0, 1.0)
        lj_btagDeepB_axis = hist.Bin('lj_btagDeepB',r"Leading jet btagDeepB" , 50, 0, 1.0)
        lj_btagDeepC_axis = hist.Bin('lj_btagDeepC',r"Leading jet btagDeepC" , 50, 0, 1.0)
        lj_btagDeepFlavB_axis = hist.Bin('lj_btagDeepFlavB',r"Leading jet btagDeepFlavB" , 50, 0, 1.0)
        lj_btagDeepFlavC_axis = hist.Bin('lj_btagDeepFlavC',r"Leading jet btagDeepFlavC" , 50, 0, 1.0)
        
        
        # Define similar axes dynamically
        disc_list = ["btagCMVA", "btagCSVV2", 'btagDeepB', 'btagDeepC', 'btagDeepFlavB', 'btagDeepFlavC',]
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d, 50, 0, 1))        
        
        #deepcsv_list = ["DeepCSV_trackDecayLenVal_0", "DeepCSV_trackDecayLenVal_1", "DeepCSV_trackDecayLenVal_2", "DeepCSV_trackDecayLenVal_3", "DeepCSV_trackDecayLenVal_4", "DeepCSV_trackDecayLenVal_5", "DeepCSV_trackDeltaR_0", "DeepCSV_trackDeltaR_1", "DeepCSV_trackDeltaR_2", "DeepCSV_trackDeltaR_3", "DeepCSV_trackDeltaR_4", "DeepCSV_trackDeltaR_5"]
        deepcsv_axes = []
        #for d in deepcsv_list:
            #if "trackDecayLenVal" in d:
                #deepcsv_axes.append(hist.Bin(d, d, 50, 0, 2.0))
            #else:
                #deepcsv_axes.append(hist.Bin(d, d, 50, 0, 0.3))

        # Define histograms from axes
        _hist_jet_dict = {
                'pt'  : hist.Hist("Counts", dataset_axis,flav_axis, jet_pt_axis),
                'eta' : hist.Hist("Counts", dataset_axis,flav_axis, jet_eta_axis),
                'phi' : hist.Hist("Counts", dataset_axis,flav_axis, jet_phi_axis),
                'mass': hist.Hist("Counts", dataset_axis,flav_axis, jet_mass_axis),
            }
        _hist_deepcsv_dict = {
                'pt'  : hist.Hist("Counts", dataset_axis,flav_axis, jet_pt_axis),
                'eta' : hist.Hist("Counts", dataset_axis, flav_axis,jet_eta_axis),
                'phi' : hist.Hist("Counts", dataset_axis, flav_axis,jet_phi_axis),
                'mass': hist.Hist("Counts", dataset_axis,flav_axis, jet_mass_axis),
            }
 
        # Generate some histograms dynamically
        for disc, axis in zip(disc_list, btag_axes):
            _hist_jet_dict[disc] = hist.Hist("Counts", dataset_axis, flav_axis,axis)
        #for deepcsv, axis in zip(deepcsv_list, deepcsv_axes):
            #_hist_deepcsv_dict[deepcsv] = hist.Hist("Counts", dataset_axis, axis)
        
        _hist_event_dict = {
                'njet'  : hist.Hist("Counts", dataset_axis, njet_axis),
                'nbjet_t' : hist.Hist("Counts", dataset_axis, nbjet_t_axis),
                'nbjet_m' : hist.Hist("Counts", dataset_axis, nbjet_m_axis),
                'nbjet_l' : hist.Hist("Counts", dataset_axis, nbjet_l_axis),
                'nel'   : hist.Hist("Counts", dataset_axis, nel_axis),
                'nmu'   : hist.Hist("Counts", dataset_axis, nmu_axis),
                'lelpt' : hist.Hist("Counts", dataset_axis, lelpt_axis),
                'lmupt' : hist.Hist("Counts", dataset_axis, lmupt_axis),
                'ljpt'  : hist.Hist("Counts", dataset_axis,flav_axis, ljpt_axis),
                'ljeta'  : hist.Hist("Counts", dataset_axis,flav_axis, ljeta_axis),
                'ljphi'  : hist.Hist("Counts", dataset_axis,flav_axis, ljphi_axis),
                'ljpt_etacut'  : hist.Hist("Counts", dataset_axis,flav_axis, ljpt_etacut_axis),
                'jet_pt_etacut'  : hist.Hist("Counts", dataset_axis,flav_axis, jet_pt_etacut_axis),
                'jet_pt_nocut'  : hist.Hist("Counts", dataset_axis,flav_axis, jet_pt_nocut_axis),
                'sljpt'  : hist.Hist("Counts", dataset_axis, sljpt_axis),
                'PV_npvsGood'  : hist.Hist("Counts", dataset_axis, PV_npvsGood_axis),
                'Pileup_nTrueInt'  : hist.Hist("Counts", dataset_axis, Pileup_nTrueInt_axis),
                'ntracks'  : hist.Hist("Counts", dataset_axis, flav_axis,ntracks_axis),
                'SV_mass'  : hist.Hist("Counts", dataset_axis,flav_axis,SV_mass_axis),
                'SV_dlen'  : hist.Hist("Counts", dataset_axis, flav_axis,SV_dlen_axis),
                'SV_dlenSig'  : hist.Hist("Counts", dataset_axis,flav_axis, SV_dlenSig_axis),
                'SV_deltaR'  : hist.Hist("Counts", dataset_axis,flav_axis, SV_deltaR_axis),
                'SV_pt'  : hist.Hist("Counts", dataset_axis,flav_axis, SV_pt_axis),
                'lj_btagCMVA'  : hist.Hist("Counts", dataset_axis,flav_axis, lj_btagCMVA_axis),
                'lj_btagCSVV2'  : hist.Hist("Counts", dataset_axis,flav_axis, lj_btagCSVV2_axis),
                'lj_btagDeepB'  : hist.Hist("Counts", dataset_axis,flav_axis, lj_btagDeepB_axis),
                'lj_btagDeepC'  : hist.Hist("Counts", dataset_axis,flav_axis, lj_btagDeepC_axis),
                'lj_btagDeepFlavB'  : hist.Hist("Counts", dataset_axis,flav_axis, lj_btagDeepFlavB_axis),
                'lj_btagDeepFlavC'  : hist.Hist("Counts", dataset_axis,flav_axis,lj_btagDeepFlavC_axis),
            }
        
        self.jet_hists = list(_hist_jet_dict.keys())
        #self.deepcsv_hists = list(_hist_deepcsv_dict.keys())
        self.event_hists = list(_hist_event_dict.keys())
    
        _hist_dict = {**_hist_jet_dict, **_hist_deepcsv_dict, **_hist_event_dict}
        self._accumulator = processor.dict_accumulator(_hist_dict)
        self._accumulator['sumw'] = processor.defaultdict_accumulator(float)


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        isData = not hasattr(events, "genWeight")
        
        if(isData):output['sumw'][dataset] += 1.
        else:output['sumw'][dataset] += ak.sum(events.genWeight)
        
        weights = Weights(len(events), storeIndividual=True)
        if not isData:
            weights.add('genweight',events.genWeight)
            print(events.genWeight)
        
        
        ##############
        # Trigger level
        triggers = [
        "HLT_PFJet140",   
        ]
        
        trig_arrs = [events.HLT[_trig.strip("HLT_")] for _trig in triggers]
        req_trig = np.zeros(len(events), dtype='bool')
        for t in trig_arrs:
            req_trig = req_trig | t
            #print(req_trig)

        ############
        # Event level
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        #events.Muon = events.Muon[(events.Muon.pt > 5)] # & (events.Muon.tightId > .5)
        #events.Muon = ak.pad_none(events.Muon, 1, axis=1) 
        #req_muon =(ak.count(events.Muon.pt, axis=1) >= 1)
        
        ## Electron cuts
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        #events.Electron = events.Electron[(events.Electron.pt > 30) & (abs(events.Electron.eta) < 2.4)]
        #events.Electron = ak.pad_none(events.Electron, 1, axis=1) 
        #req_ele = (ak.count(events.Electron.pt, axis=1) >= 1)
        
        ## Jet cuts
        #events.Jet = events.Jet[(abs(events.Jet.eta) < 2.4)&(events.Jet.pt > 180)]#events.Jet[(events.Jet.pt > 40) & (events.Jet.pt < 250)]
        events_Jet = events.Jet[(abs(events.Jet.eta) < 2.4)&(events.Jet.pt > 180)]
        req_jets = (ak.count(events_Jet.pt, axis=1) >= 1)#at least one jet in the event pass the pt and eta cut
        #if want every jet in the event to pass pt and eta cuts, req_jets = (events_Jet)
        
        
        #Jet without pT cut
        Jet_all = events.Jet[(abs(events.Jet.eta) < 2.4)]
        req_jets_all = (ak.count(Jet_all.pt, axis=1) >= 1)  

        
        #req_opposite_charge = events.Electron[:, 0].charge * events.Muon[:, 0].charge == -1
        
        event_level = (req_trig&req_jets)
         
        
        # Selected
        selev = events[event_level]
        

        
        
        #########
        
        # Per electron
        #el_eta   = (abs(selev.Electron.eta) <= 2.4)
        #el_pt    = selev.Electron.pt > 30
        #el_level = el_eta & el_pt
        
        # Per muon
        #mu_eta   = (abs(selev.Muon.eta) <= 2.4)
        #mu_pt    = (selev.Muon.pt > 5)
        #mu_level =  mu_pt
        
        # Per jet
        jet_eta    = (abs(selev.Jet.eta) < 2.4)
        jet_pt     = (selev.Jet.pt > 180) #& (selev.Jet.pt < 250)
        #jet_pu     = ( ((selev.Jet.puId > 6) & (selev.Jet.pt < 50)) | (selev.Jet.pt > 50) ) 
        #jet_id     = selev.Jet.jetId >= 2 
        #jet_id     = selev.Jet.isTight() == 1 & selev.Jet.isTightLeptonVeto() == 0
        jet_level  =  (jet_eta&jet_pt) 
        
        #jet without pT cut
        
        jet_all_level= jet_eta 
        

        # b-tag twiki : https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
        bjet_disc_t  = selev.Jet.btagDeepB > 0.7264 # L=0.0494, M=0.2770, T=0.7264
        bjet_disc_m  = selev.Jet.btagDeepB > 0.2770 # L=0.0494, M=0.2770, T=0.7264
        bjet_disc_l  = selev.Jet.btagDeepB > 0.0494 # L=0.0494, M=0.2770, T=0.7264
        bjet_level_t = jet_level & bjet_disc_t
        bjet_level_m = jet_level & bjet_disc_m
        bjet_level_l = jet_level & bjet_disc_l
        
        
        
        
        #print('jetIdx', len(selev.JetSVs),selev.JetSVs.jetIdx)
        #print('jetIdx not empty', ak.num(selev.JetSVs), ak.num(selev.JetSVs, axis=-1) > 0)
        #print('sjets', len(selev.Jet), ak.num(selev.JetSVs.jetIdx))


        #print('sjets', ak.count(selev.Jet[:,0].pt),ak.count(selev.Jet.pt))
        
        #print('matched jets', selev.Jet[selev.JetSVs.jetIdx])
       
        #sel    = selev.Electron[el_level]
        #smu    = selev.Muon[mu_level]
        sjets  = selev.Jet[jet_level]
        
        #check_sjets = (ak.count(sjets.pt,axis = 1)>0) 
        #sjets = sjets[check_sjets]
        
        
        sjets_all  = selev.Jet[jet_all_level] # jets without pT cuts
        #JetSVs
        
        #pairing
        #pairs = ak.cartesian([sjets[:,0], selev.SV])
        #check if pair exists
        #check_pairs = np.where(ak.count(pairs.slot0.pt,axis=-1)==0,False,True)
        #keep pairs if not empty
        #pairs =pairs[check_pairs]
        #Distance between SV and Jet
        #raw_match=pairs.slot0.delta_r(pairs.slot1)

        #select the index of closest raw_match
        #mindR_index = ak.argmin(raw_match,axis=-1)

        #closest dR 
        #min_dR = raw_match[np.arange(ak.size(raw_match,0)),ak.to_numpy(mindR_index)]

        #deltaR cut <0.4
        #isitassociatedtoSV = min_dR <0.4
        #choose pairs with closest delta R
        #jet_and_closest_SV = pairs.slot0[np.arange(ak.size(pairs.slot0,0)),ak.to_numpy(mindR_index)]

        #SV closest to jet
        #closest_SV = pairs.slot1[np.arange(ak.size(pairs.slot1,0)),ak.to_numpy(mindR_index)]

        #Finally!!!!
        #jet_associated_SV = jet_and_closest_SV[isitassociatedtoSV]        
        #Selected_SV = closest_SV[isitassociatedtoSV]
        #dR = min_dR[isitassociatedtoSV]
        

        
        #matched_JetSVs=  selev.Jet[selev.JetSVs.jetIdx]
            #matched_JetSVs = matched_JetSVs[ak.any(jet_level,axis=-1)]
            #lj_matched_JetSVs = matched_JetSVs[selev.JetSVs.jetIdx==0]
            #lj_SVs = selev.JetSVs[selev.JetSVs.jetIdx==0]

            
           
            
        #print('jetSVs', ak.count(selev.JetSVs.pt))
        #print('jet_level', ak.count(jet_level))
        matched_JetSVs=  selev.Jet[selev.JetSVs.jetIdx]
        matched_JetSVs = matched_JetSVs[ak.any(jet_level,axis=-1)]
        
        #print('matched_JetSVs',matched_JetSVs,ak.count(matched_JetSVs))
        #print('jet level',jet_level)
        
        #print('jetIdx', selev.JetSVs.jetIdx)
        #print('sjets', ak.count(sjets[:,0].pt),ak.count(sjets.pt))
        #print('matched_JetIdx',ak.count(matched_JetSVs.pt), matched_JetSVs)
        #print('leading jet', matched_JetSVs[selev.JetSVs.jetIdx==0])
        
        
        lj_matched_JetSVs = matched_JetSVs[selev.JetSVs.jetIdx==0]
        lj_SVs = selev.JetSVs[selev.JetSVs.jetIdx==0]
        #print(len(lj_matched_JetSVs),len(sjets),len(sjets[:,0]))
        
        

        
        sbjets_t = selev.Jet[bjet_level_t]
        sbjets_m = selev.Jet[bjet_level_m]
        sbjets_l = selev.Jet[bjet_level_l]
        
        
        if isData:
            genflavor = ak.zeros_like(sjets.pt)
        else:
            genflavor = sjets.hadronFlavour
        
        # output['pt'].fill(dataset=dataset, pt=selev.Jet.pt.())
        # Fill histograms dynamically  
        for histname, h in output.items():
            if (histname not in self.jet_hists): continue#and (histname not in self.deepcsv_hists): continue
            # Get valid fields perhistogram to fill
            fields = {k: ak.flatten(sjets[k], axis=None) for k in h.fields if k in dir(sjets)}
            genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],sjets['pt'])[0])
            h.fill(dataset=dataset,flav=ak.flatten(genflavor), **fields,weight=genweiev)


        def flatten(ar): # flatten awkward into a 1d array to hist
            return ak.flatten(ar, axis=None)

        def num(ar):
            return ak.num(ak.fill_none(ar[~ak.is_none(ar)], 0), axis=0)

        output['njet'].fill(dataset=dataset, njet=flatten(ak.num(sjets)))
        
        output['nbjet_t'].fill(dataset=dataset, nbjet_t=flatten(ak.num(sbjets_t)))
        output['nbjet_m'].fill(dataset=dataset, nbjet_m=flatten(ak.num(sbjets_m)))
        output['nbjet_l'].fill(dataset=dataset, nbjet_l=flatten(ak.num(sbjets_l)))
        #output['nel'].fill(dataset=dataset,   nel=flatten(ak.num(sel)))
        #output['nmu'].fill(dataset=dataset,   nmu=flatten(ak.num(smu)))

        #output['lelpt'].fill(dataset=dataset, lelpt=flatten(selev.Electron[:, 0].pt))
        #output['lmupt'].fill(dataset=dataset, lmupt=flatten(selev.Muon[:, 0].pt))
        #output['ljpt'].fill(dataset=dataset,  ljpt=flatten(selev.Jet[:, 0].pt))

        if isData: 
            output['ljpt'].fill(dataset=dataset, flav=0,ljpt=flatten(sjets[:, 0].pt))
            output['ljeta'].fill(dataset=dataset, flav=0, ljeta=flatten(sjets[:, 0].eta))
            output['ljphi'].fill(dataset=dataset, flav=0, ljphi=flatten(sjets[:, 0].phi))
            output['ljpt_etacut'].fill(dataset=dataset,flav=0,  ljpt_etacut=flatten(sjets_all[:, 0].pt))
            output['jet_pt_etacut'].fill(dataset=dataset,flav=0, jet_pt_etacut=flatten(sjets_all.pt))
            output['jet_pt_nocut'].fill(dataset=dataset,flav=0, jet_pt_nocut=flatten(events.Jet.pt))
            
            #discriminators
            output['lj_btagCMVA'].fill(dataset=dataset,flav=0, lj_btagCMVA=flatten(sjets[:, 0].btagCMVA))
            output['lj_btagCSVV2'].fill(dataset=dataset, flav=0, lj_btagCSVV2=flatten(sjets[:, 0].btagCSVV2))
            output['lj_btagDeepB'].fill(dataset=dataset,flav=0, lj_btagDeepB=flatten(sjets[:, 0].btagDeepB))
            output['lj_btagDeepC'].fill(dataset=dataset,flav=0, lj_btagDeepC=flatten(sjets[:, 0].btagDeepC))
            output['lj_btagDeepFlavB'].fill(dataset=dataset, flav=0, lj_btagDeepFlavB=flatten(sjets[:, 0].btagDeepFlavB))
            output['lj_btagDeepFlavC'].fill(dataset=dataset, flav=0, lj_btagDeepFlavC=flatten(sjets[:, 0].btagDeepFlavC))
            #SVs
            output['ntracks'].fill(dataset=dataset,flav=0, ntracks=flatten(lj_SVs.ntracks))
            output['SV_mass'].fill(dataset=dataset, flav=0, SV_mass=flatten(lj_SVs.mass))
            #output['SV_dlen'].fill(dataset=dataset,flav=0, SV_dlen=flatten(Selected_SV.dlen))
            #output['SV_dlenSig'].fill(dataset=dataset,flav=0, SV_dlenSig=flatten(Selected_SV.dlenSig))
            output['SV_deltaR'].fill(dataset=dataset,flav=0, SV_deltaR=flatten(lj_SVs.deltaR))
            output['SV_pt'].fill(dataset=dataset,flav=0, SV_pt=abs(flatten(lj_SVs.pt)))
            
        else: 
            output['ljpt'].fill(dataset=dataset, flav=flatten(sjets[:,0].hadronFlavour),ljpt=flatten(sjets[:, 0].pt))
            output['ljeta'].fill(dataset=dataset, flav=flatten(sjets[:,0].hadronFlavour), ljeta=flatten(sjets[:, 0].eta))
            output['ljphi'].fill(dataset=dataset, flav=flatten(sjets[:,0].hadronFlavour), ljphi=flatten(sjets[:, 0].phi))
            output['ljpt_etacut'].fill(dataset=dataset,flav=flatten(sjets_all[:, 0].hadronFlavour),  ljpt_etacut=flatten(sjets_all[:, 0].pt))
            output['jet_pt_etacut'].fill(dataset=dataset,flav=flatten(sjets_all.hadronFlavour), jet_pt_etacut=flatten(sjets_all.pt))
            output['jet_pt_nocut'].fill(dataset=dataset,flav=flatten(events.Jet.hadronFlavour), jet_pt_nocut=flatten(events.Jet.pt))
            
            #discriminators
            output['lj_btagCMVA'].fill(dataset=dataset,flav=flatten(sjets[:,0].hadronFlavour), lj_btagCMVA=flatten(sjets[:, 0].btagCMVA))
            output['lj_btagCSVV2'].fill(dataset=dataset, flav=flatten(sjets[:,0].hadronFlavour), lj_btagCSVV2=flatten(sjets[:, 0].btagCSVV2))
            output['lj_btagDeepB'].fill(dataset=dataset,flav=flatten(sjets[:,0].hadronFlavour), lj_btagDeepB=flatten(sjets[:, 0].btagDeepB))
            output['lj_btagDeepC'].fill(dataset=dataset,flav=flatten(sjets[:,0].hadronFlavour), lj_btagDeepC=flatten(sjets[:, 0].btagDeepC))
            output['lj_btagDeepFlavB'].fill(dataset=dataset, flav=flatten(sjets[:,0].hadronFlavour), lj_btagDeepFlavB=flatten(sjets[:, 0].btagDeepFlavB))
            output['lj_btagDeepFlavC'].fill(dataset=dataset, flav=flatten(sjets[:,0].hadronFlavour), lj_btagDeepFlavC=flatten(sjets[:, 0].btagDeepFlavC))
            #SV
            #print(ak.count(sjets.hadronFlavour), ak.count(JetSVs.jetIdx))
 
            #print (flatten(sjets[:,0][JetSVs.jetidx]),ak.count(sjets[:,0][JetSVs.jetidx]))
            output['ntracks'].fill(dataset=dataset,flav=flatten(lj_matched_JetSVs.hadronFlavour), ntracks=flatten(lj_SVs.ntracks))
            output['SV_mass'].fill(dataset=dataset, flav=flatten(lj_matched_JetSVs.hadronFlavour), SV_mass=flatten(lj_SVs.mass))
            #output['SV_dlen'].fill(dataset=dataset,flav=flatten(jet_associated_SV.hadronFlavour), SV_dlen=flatten(Selected_SV.dlen))
            #output['SV_dlenSig'].fill(dataset=dataset,flav=flatten(jet_associated_SV.hadronFlavour), SV_dlenSig=flatten(Selected_SV.dlenSig))
            output['SV_deltaR'].fill(dataset=dataset,flav=flatten(lj_matched_JetSVs.hadronFlavour), SV_deltaR=flatten(lj_SVs.deltaR))
            output['SV_pt'].fill(dataset=dataset,flav=flatten(lj_matched_JetSVs.hadronFlavour), SV_pt=abs(flatten(lj_SVs.pt)))
            output['Pileup_nTrueInt'].fill(dataset=dataset,  Pileup_nTrueInt=flatten(events.Pileup.nTrueInt))
            
            
        #PV
        output['PV_npvsGood'].fill(dataset=dataset,  PV_npvsGood=flatten(events.PV.npvsGood))
        
        
        #print(flatten(lj_SVs.deltaR))
        
        #output['sljpt'].fill(dataset=dataset,  sljpt=flatten(selev.Jet[:, 1].pt))
        
        return output

    def postprocess(self, accumulator):
        return accumulator
