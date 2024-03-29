MMR = ['MLH1', 'MLH3', 'PMS1', 'PMS2', 'MSH2', 'MSH3', 'MSH6']
HR = [
    'BRCA1', 'BRCA2', 'ATM', 'BARD1', 'BRIP1', 'CDK12', 'CHEK1', 'CHEK2',
    'FANCL', 'PALB2', 'PPP2R2A', 'RAD51B', 'RAD51C', 'RAD51D', 'RAD54L'
]

MET_14_skipping = [
    'c.3028+1G>T', 'c.2888-29_2888-6del24', 'c.3028+2_3028+17del',
    'c.3028+2_3028+6del', 'c.3028+2T>C', 'c.3009_3028+3del',
    'c.3028_3028+15del', 'c.3028+2T>A', 'c.2888-1G>C', 'c.2888-25_2900del',
    'c.2888-29_2888-2del', 'c.2888-2A>C', 'c.2888-31_2889del',
    'c.2888-40_2889del', 'c.2888-8_2913del', 'c.2942-18_2945del',
    'c.3015_3028+7delinsC', 'c.3018_3028+4del', 'c.3028+1_3028+10del',
    'c.3028+1G>A', 'c.3028+1G>C', 'c.2888-35_2888del', 'c.2888-7_2920del',
    'c.2888-2_2915del', 'c.2888-22_2888-2del', 'c.2888-20_2888-1del',
    'c.3018_3028+8del', 'c.3028+1_3028+9del', 'c.3010_3028+8del',
    'c.2903_3028+67del', 'c.3020_3028+24del', 'c.2888-20_2939del',
    'c.2888-28_2888-14del', 'c.2888-21_2888-5del', 'c.2888-19_2888-3del',
    'c.2905_2940del', 'c.2888-35_2888-17del', 'c.2888-18_2888-7del12',
    'c.3024_3028+7del12'
]

genes_688 = [
    'ABCB1', 'ABCG2', 'ABL1', 'ABRAXAS1', 'ACSL3', 'ACVR1', 'ACVR2A', 'ACYP2',
    'ADGRA2', 'AFF4', 'AJUBA', 'AKT1', 'AKT2', 'AKT3', 'ALK', 'AMER1', 'APC',
    'APOB', 'AR', 'ARAF', 'ARID1A', 'ARID1B', 'ARID2', 'ASXL1', 'ATAD2',
    'ATF1', 'ATM', 'ATR', 'ATRX', 'AURKA', 'AURKB', 'AXIN1', 'AXIN2', 'AXL',
    'B2M', 'BABAM2', 'BACH1', 'BAP1', 'BARD1', 'BCL2', 'BCL2A1', 'BCL2L1',
    'BCL6', 'BCOR', 'BCR', 'BIRC2', 'BIRC3', 'BLM', 'BMPR1A', 'BRAF', 'BRCA1',
    'BRCA2', 'BRCC3', 'BRD4', 'BRF1', 'BRIP1', 'BTK', 'C8orf34', 'CARD11',
    'CARM1', 'CASP8', 'CASR', 'CBL', 'CBLB', 'CBR3', 'CBX4', 'CCDC6', 'CCNA2',
    'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CD74', 'CD79B', 'CD274', 'CD276',
    'CDC27', 'CDC42', 'CDC73', 'CDH1', 'CDH9', 'CDK4', 'CDK6', 'CDK8', 'CDK12',
    'CDKN1A', 'CDKN1B', 'CDKN1C', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'CDRT4',
    'CDX2', 'CEBPA', 'CETN2', 'CFTR', 'CHD1', 'CHEK1', 'CHEK2', 'CIC', 'CLK2',
    'COL11A1', 'COL22A1', 'COP1', 'CREB1', 'CREBBP', 'CRKL', 'CSDE1', 'CSF1R',
    'CSMD3', 'CTCF', 'CTLA4', 'CTNNA1', 'CTNNB1', 'CTNND2', 'CUL3', 'CUL4A',
    'CUL4B', 'CXCR4', 'CYLD', 'CYP2C8', 'CYP2D6', 'CYP11B1', 'CYP17A1',
    'CYP19A1', 'DAXX', 'DCUN1D1', 'DDB2', 'DDR1', 'DDR2', 'DICER1', 'DIS3',
    'DMC1', 'DNMT3A', 'DNTT', 'DOCK2', 'DOT1L', 'DPYD', 'DSCAM', 'DUSP4',
    'DUT', 'DYNC2H1', 'E2F3', 'EDC4', 'EGFR', 'EIF1AX', 'EIF4A2', 'ELAC2',
    'ELF3', 'ELOC', 'EME1', 'EME2', 'EML4', 'EMSY', 'EP300', 'EPCAM', 'EPHA2',
    'EPHA3', 'EPHA4', 'EPHB1', 'EPPK1', 'ERBB2', 'ERBB3', 'ERBB4', 'ERCC1',
    'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'ERCC6', 'ERF', 'ERG', 'ERRFI1',
    'ESR1', 'ETV1', 'ETV4', 'ETV5', 'ETV6', 'EWSR1', 'EXO1', 'EXOC2', 'EXT1',
    'EXT2', 'EZH1', 'EZH2', 'EZR', 'FAM135B', 'FAN1', 'FANCA', 'FANCB',
    'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 'FANCI', 'FANCL', 'FANCM',
    'FAT1', 'FAT2', 'FAT3', 'FAT4', 'FBXW7', 'FCGR2B', 'FCGR3A', 'FGD4',
    'FGF2', 'FGF3', 'FGF4', 'FGF6', 'FGF10', 'FGF12', 'FGF14', 'FGF19',
    'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'FH', 'FLCN', 'FLI1', 'FLNA', 'FLT1',
    'FLT3', 'FLT4', 'FOXA1', 'FOXL2', 'FOXO1', 'FOXP1', 'FRAS1', 'FUBP1',
    'FYN', 'G6PC', 'GAB2', 'GABRA6', 'GALNT12', 'GATA1', 'GATA2', 'GATA3',
    'GATA4', 'GATA6', 'GEN1', 'GGH', 'GID4', 'GLI1', 'GNA11', 'GNAQ', 'GNAS',
    'GPS2', 'GRB7', 'GREM1', 'GRIN2A', 'GRM3', 'GSK3B', 'GSTP1', 'H1-2',
    'H2AX', 'H2BC5', 'H3-3A', 'H3-3B', 'H3-4', 'H3C1', 'H3C2', 'H3C3', 'H3C4',
    'H3C6', 'H3C7', 'H3C8', 'H3C10', 'H3C11', 'H3C13', 'H3C14', 'HDAC1', 'HGF',
    'HLA-A', 'HLA-B', 'HNF1A', 'HOXB13', 'HRAS', 'HSD3B1', 'HSD17B4',
    'HSP90AA1', 'HSPA4', 'ICOSLG', 'ID3', 'IDH1', 'IDH2', 'IFNGR1', 'IGF1',
    'IGF1R', 'IGF2', 'IGF2R', 'IKBKE', 'IKZF1', 'IL7R', 'IL10', 'INHA',
    'INHBA', 'INPP4A', 'INPP4B', 'INSR', 'IRF2', 'IRF4', 'IRS2', 'JAK1',
    'JAK2', 'JAK3', 'JMJD1C', 'JUN', 'KDM5C', 'KDM6A', 'KDR', 'KEAP1',
    'KIAA1549', 'KIF1B', 'KIF5B', 'KIT', 'KLF6', 'KLHL6', 'KLLN', 'KMT2A',
    'KMT2B', 'KMT2C', 'KMT2D', 'KMT5A', 'KNSTRN', 'KRAS', 'LAMA2', 'LATS1',
    'LATS2', 'LHCGR', 'LIFR', 'LIG4', 'LRP1B', 'LRRK1', 'LRRK2', 'LTK', 'LYN',
    'LZTR1', 'MALT1', 'MAP2K1', 'MAP2K2', 'MAP2K4', 'MAP3K1', 'MAP3K4',
    'MAP3K13', 'MAP3K14', 'MAP4K3', 'MAPK1', 'MAPK3', 'MAPKAP1', 'MAX',
    'MB21D2', 'MC1R', 'MCL1', 'MDC1', 'MDH2', 'MDM2', 'MDM4', 'MECOM', 'MED12',
    'MEF2B', 'MEN1', 'MERTK', 'MET', 'MGA', 'MGMT', 'MITF', 'MKNK1', 'MLH1',
    'MLH3', 'MMS19', 'MPL', 'MRE11', 'MS4A1', 'MSH2', 'MSH3', 'MSH4', 'MSH5',
    'MSH6', 'MSI1', 'MSI2', 'MST1', 'MST1R', 'MTAP', 'MTDH', 'MTHFR', 'MTOR',
    'MTRR', 'MUC6', 'MUC16', 'MUS81', 'MUTYH', 'MYB', 'MYC', 'MYCL', 'MYCN',
    'MYD88', 'MYOD1', 'MYSM1', 'NABP2', 'NBN', 'NCOA2', 'NCOA3', 'NCOA4',
    'NCOR1', 'NCOR2', 'NEGR1', 'NEIL2', 'NF1', 'NF2', 'NFE2L2', 'NFKB1',
    'NFKBIA', 'NHEJ1', 'NKX2-1', 'NKX3-1', 'NLRP1', 'NOTCH1', 'NOTCH2',
    'NOTCH3', 'NOTCH4', 'NPM1', 'NQO1', 'NR4A3', 'NRAS', 'NSD1', 'NSD2',
    'NSD3', 'NT5C2', 'NTHL1', 'NTRK1', 'NTRK2', 'NTRK3', 'NUDT18', 'NUF2',
    'NUTM1', 'NYAP2', 'PAK1', 'PAK5', 'PALB2', 'PARP1', 'PARP2', 'PARP3',
    'PARP4', 'PAX5', 'PAX8', 'PBRM1', 'PBX1', 'PCDH9', 'PDCD1', 'PDCD1LG2',
    'PDGFRA', 'PDGFRB', 'PDK1', 'PGR', 'PHF6', 'PHOX2B', 'PIK3CA', 'PIK3CB',
    'PIK3CG', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PIM1', 'PLAG1', 'PLCG2', 'PLK1',
    'PLK2', 'PLXNA1', 'PMAIP1', 'PMS1', 'PMS2', 'PNPLA3', 'PNRC1', 'POLD1',
    'POLE', 'POLG', 'POLH', 'POLM', 'POLN', 'POLQ', 'POT1', 'POU5F1', 'PPARG',
    'PPM1D', 'PPP2R1A', 'PPP2R2A', 'PPP4R2', 'PPP6C', 'PRDM1', 'PRDM14',
    'PREX2', 'PRKAR1A', 'PRKCI', 'PRKD1', 'PRKDC', 'PRKN', 'PRPF40B', 'PRSS1',
    'PTCH1', 'PTCH2', 'PTEN', 'PTGIS', 'PTP4A1', 'PTPN11', 'PTPRD', 'PTPRO',
    'PTPRS', 'PTPRT', 'QKI', 'RAB35', 'RAC1', 'RAC2', 'RAD21', 'RAD50',
    'RAD51', 'RAD51B', 'RAD51C', 'RAD51D', 'RAD52', 'RAD54B', 'RAD54L', 'RAF1',
    'RARA', 'RASA1', 'RB1', 'RBBP8', 'RBM10', 'RECQL', 'RECQL4', 'REEP5',
    'REL', 'RET', 'RFC4', 'RHEB', 'RHOA', 'RICTOR', 'RIT1', 'RNF43', 'ROS1',
    'RPS6KA3', 'RPS6KA4', 'RPS6KB2', 'RRAGC', 'RRAS', 'RRAS2', 'RSPO2',
    'RTEL1', 'RUFY4', 'RUNX1', 'RXRA', 'RYBP', 'RYR2', 'RYR3', 'SCG5', 'SDC4',
    'SDHA', 'SDHAF2', 'SDHB', 'SDHC', 'SDHD', 'SEMA3C', 'SESN1', 'SESN2',
    'SESN3', 'SETD2', 'SF3B1', 'SGK1', 'SH2B3', 'SH2D1A', 'SHOC2', 'SHPRH',
    'SHQ1', 'SIPA1', 'SLC7A8', 'SLC28A3', 'SLC34A2', 'SLC45A3', 'SLCO1B1',
    'SLX1A', 'SLX4', 'SMAD2', 'SMAD3', 'SMAD4', 'SMARCA1', 'SMARCA4',
    'SMARCB1', 'SMARCD1', 'SMO', 'SMYD3', 'SNCAIP', 'SOCS1', 'SOD2', 'SOS1',
    'SOX2', 'SOX4', 'SOX9', 'SOX10', 'SOX17', 'SPEN', 'SPINK1', 'SPOP',
    'SPOPL', 'SPRED1', 'SRC', 'SRSF2', 'STAG1', 'STAG2', 'STAT3', 'STAT5A',
    'STAT5B', 'STK11', 'STK19', 'STK40', 'SUFU', 'SUZ12', 'SYK', 'TAF1L',
    'TAF15', 'TAP1', 'TAP2', 'TBL1XR1', 'TBX3', 'TCF3', 'TCF4', 'TCF7L2',
    'TEK', 'TERT', 'TET1', 'TET2', 'TFE3', 'TGFBR1', 'TGFBR2', 'TIPARP',
    'TMEM127', 'TMPRSS2', 'TNFAIP3', 'TNFRSF14', 'TNFSF11', 'TOP1', 'TOP3A',
    'TOPBP1', 'TP53', 'TP53BP1', 'TP63', 'TPM3', 'TRAF2', 'TRAF7', 'TRRAP',
    'TSC1', 'TSC2', 'TSHR', 'TUBB3', 'TYMS', 'U2AF1', 'UGT1A1', 'UMPS',
    'UNC5D', 'UPF1', 'USP6', 'VEGFA', 'VHL', 'VTCN1', 'WEE1', 'WRN', 'WT1',
    'WWTR1', 'XIAP', 'XPA', 'XPC', 'XPO1', 'XRCC1', 'XRCC2', 'XRCC3', 'YAP1',
    'YES1', 'YWHAZ', 'ZBTB16', 'ZFHX3', 'ZFHX4', 'ZMYM3', 'ZNF2', 'ZNF217',
    'ZNF703', 'ZNF770', 'ZNRF3', 'ZRSR2'
]