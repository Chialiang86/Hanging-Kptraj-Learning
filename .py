import shutil, os

src_paths = [
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_0_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_101_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_104_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_105_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_106_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_108_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_110_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_112_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_113_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_118_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_134_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_136_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_137_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_138_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_140_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_142_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_143_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_147_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_148_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_149_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_154_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_158_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_167_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_16_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_172_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_176_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_180_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_181_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_183_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_189_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_18_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_190_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_191_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_198_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_199_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_205_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_206_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_207_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_218_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_221_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_227_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_22_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_230_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_232_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_237_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_238_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_241_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_243_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_245_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_249_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_253_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_266_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_267_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_270_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_276_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_279_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_286_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_290_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_293_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_296_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_300_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_303_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_306_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_309_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_313_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_314_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_333_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_335_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_338_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_340_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_347_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_34_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_351_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_354_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_356_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_359_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_362_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_363_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_364_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_376_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_380_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_386_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_38_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_390_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_398_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_399_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_3_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_40_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_45_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_46_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_50_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_52_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_54_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_62_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_63_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_65_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_69_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_74_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_75_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_86_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_89_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hcu_8_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_103_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_105_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_109_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_10_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_110_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_117_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_11_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_120_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_121_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_124_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_126_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_134_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_137_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_142_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_148_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_152_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_154_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_156_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_157_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_166_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_170_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_174_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_176_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_177_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_178_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_180_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_190_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_191_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_193_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_196_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_198_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_203_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_205_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_206_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_207_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_208_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_210_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_216_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_219_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_21_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_223_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_229_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_230_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_232_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_233_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_235_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_236_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_23_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_251_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_252_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_255_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_25_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_260_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_266_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_275_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_277_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_279_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_286_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_287_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_28_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_290_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_291_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_293_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_294_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_304_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_30_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_310_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_311_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_314_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_315_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_317_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_321_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_326_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_327_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_329_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_331_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_338_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_339_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_342_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_347_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_348_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_349_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_350_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_354_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_35_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_361_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_362_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_363_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_364_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_370_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_374_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_382_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_390_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_391_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_393_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_396_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_397_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_398_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_42_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_43_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_46_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_48_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_50_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_52_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_54_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_55_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_58_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_60_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_70_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_71_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_74_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_75_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_7_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_80_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_81_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_83_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_86_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_92_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_94_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hs_95_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_0_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_101_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_102_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_104_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_107_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_109_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_114_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_115_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_117_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_118_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_11_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_120_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_123_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_125_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_127_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_129_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_12_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_130_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_134_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_135_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_137_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_13_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_142_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_143_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_146_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_151_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_154_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_156_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_15_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_163_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_164_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_165_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_166_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_168_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_170_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_171_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_172_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_173_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_188_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_18_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_191_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_192_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_193_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_194_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_198_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_200_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_205_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_207_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_208_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_211_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_213_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_215_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_217_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_218_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_224_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_228_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_22_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_231_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_238_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_241_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_242_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_243_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_246_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_247_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_24_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_250_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_254_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_256_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_258_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_259_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_260_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_261_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_262_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_263_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_268_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_270_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_274_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_275_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_278_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_27_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_281_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_282_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_283_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_284_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_286_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_298_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_2_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_300_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_304_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_306_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_308_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_312_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_313_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_315_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_318_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_321_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_323_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_325_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_330_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_335_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_339_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_345_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_34_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_350_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_361_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_363_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_367_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_371_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_380_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_381_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_385_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_390_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_391_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_41_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_44_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_45_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_56_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_5_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_6_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_70_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_71_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_72_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_80_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_8_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_95_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_97_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_hsr_98_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_my_180_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_my_45_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_my_60_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_my_90_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_my_bar_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_122_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_124_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_12_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_136_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_145_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_1_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_209_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_2_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_35_easy/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_40_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_42_hard/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_44_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_47_normal/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_57_devil/affordance-fullview.npy',
    '../Hanging-Motion-Planning/models/hook_all_new/Hook_omni_84_easy/affordance-fullview.npy'
]

target_root = 'shapes/hook_all_new'
for src_path in src_paths:
    hook_name = src_path.split('/')[-2]
    path = src_path.split('/')[-1]
    dst_path = f'{target_root}/{hook_name}/{path}'
    shutil.copy2(src_path, dst_path)