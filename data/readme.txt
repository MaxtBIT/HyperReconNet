==============================================================================
A Few Descriptions of Datasets
==============================================================================
1. How to make datasets?
For training, trainsets and validsets are generated from the same data in a ratio of four to one. 
First, read the complete images from .mat files and normalize them to [0, 1]. Then, obtain the 
patches according to a given stride and save them in HDF5 files. Note that, each .h5 file should 
be in the same length and the patches have been shuffled.
For testing, the stride used for obtaining patches is half of the patch size. That is, adjacent 
patches have half overlap. When interfacing the model, the overlapping area shoud be averaged.
Afterwards, the stitched images are used to calculate the performance metrics.
==============================================================================
2. How to split datasets?
For ICVL dataset, 100 scenes are used for training, and 50 scenes are used for testing. These images
are published in http://icvl.cs.bgu.ac.il/hyperspectral/. The split filenames are as follows:
---------ICVL_train---------
001: 4cam_0411-1640-1
002: 4cam_0411-1648
003: BGU_0403-1419-1
004: BGU_0522-1113-1
005: BGU_0522-1127
006: BGU_0522-1136
007: BGU_0522-1201
008: BGU_0522-1203
009: BGU_0522-1211
010: BGU_0522-1216
011: BGU_0522-1217
012: CC_40D_2_1103-0917
013: Flower_0325-1336
014: IDS_COLORCHECK_1020-1215-1
015: IDS_COLORCHECK_1020-1223
016: Labtest_0910-1502
017: Labtest_0910-1504
018: Labtest_0910-1506
019: Labtest_0910-1509
020: Labtest_0910-1510
021: Labtest_0910-1511
022: Labtest_0910-1513
023: Lehavim_0910-1622
024: Lehavim_0910-1630
025: Lehavim_0910-1633
026: Lehavim_0910-1635
027: Lehavim_0910-1636
028: Lehavim_0910-1640
029: Lehavim_0910-1708
030: Lehavim_0910-1716
031: Lehavim_0910-1717
032: Lehavim_0910-1718
033: Lehavim_0910-1725
034: Master20150112_f2_colorchecker
035: Master2900k
036: Master5000K
037: Master5000K_2900K
038: Maz0326-1038
039: bguCAMP_0514-1723
040: bguCAMP_0514-1724
041: bgu_0403-1439
042: bgu_0403-1444
043: bgu_0403-1459
044: hill_0325-1242
045: lehavim_0910-1600
046: lehavim_0910-1602
047: lst_0408-0950
048: lst_0408-1004
049: lst_0408-1012
050: maz_0326-1048
051: objects_0924-1652
052: omer_0331-1055
053: omer_0331-1102
054: omer_0331-1104
055: omer_0331-1118
056: omer_0331-1119
057: omer_0331-1130
058: omer_0331-1131
059: omer_0331-1135
060: omer_0331-1150
061: omer_0331-1159
062: pepper_0503-1228
063: pepper_0503-1229
064: pepper_0503-1236
065: peppers_0503-1308
066: peppers_0503-1311
067: peppers_0503-1315
068: peppers_0503-1330
069: peppers_0503-1332
070: plt_0411-1037
071: plt_0411-1046
072: plt_0411-1116
073: plt_0411-1155
074: plt_0411-1200-1
075: plt_0411-1207
076: plt_0411-1210
077: plt_0411-1211
078: plt_0411-1232-1
079: prk_0328-0945
080: prk_0328-1025
081: prk_0328-1031
082: prk_0328-1034
083: prk_0328-1037
084: prk_0328-1045
085: ramot_0325-1322
086: rmt_0328-1241-1
087: rmt_0328-1249-1
088: rsh2_0406-1505
089: rsh_0406-1343
090: rsh_0406-1356
091: rsh_0406-1413
092: rsh_0406-1427
093: rsh_0406-1441-1
094: rsh_0406-1443
095: sami_0331-1019
096: sat_0406-1107
097: sat_0406-1129
098: sat_0406-1130
099: sat_0406-1157-1
100: strt_0331-1027

---------ICVL_test---------
001: Lehavim_0910-1626
002: Lehavim_0910-1627
003: Lehavim_0910-1629
004: Ramot0325-1364
005: bguCAMP_0514-1659
006: bguCAMP_0514-1711
007: bguCAMP_0514-1712
008: bguCAMP_0514-1718
009: bgu_0403-1511
010: bgu_0403-1523
011: bgu_0403-1525
012: bulb_0822-0903
013: bulb_0822-0909
014: eve_0331-1549
015: eve_0331-1551
016: eve_0331-1601
017: eve_0331-1602
018: eve_0331-1606
019: eve_0331-1618
020: eve_0331-1632
021: eve_0331-1633
022: eve_0331-1646
023: eve_0331-1647
024: eve_0331-1656
025: eve_0331-1657
026: eve_0331-1702
027: eve_0331-1705
028: gavyam_0823-0930
029: gavyam_0823-0933
030: gavyam_0823-0944
031: gavyam_0823-0945
032: gavyam_0823-0950-1
033: grf_0328-0949
034: hill_0325-1219
035: hill_0325-1228
036: hill_0325-1235
037: lehavim_0910-1605
038: lehavim_0910-1607
039: lehavim_0910-1610
040: mor_0328-1209-2
041: nachal_0823-1038
042: nachal_0823-1040
043: nachal_0823-1047
044: nachal_0823-1110
045: nachal_0823-1117
046: nachal_0823-1118
047: nachal_0823-1121
048: nachal_0823-1127
049: nachal_0823-1132
050: nachal_0823-1144

For Havrad dataset, 35 scenes are used for training, and 9 scenes are used for testing. These images
are published in http://vision.seas.harvard.edu/hyperspec/. The split filenames are as follows:
---------Havrad_train---------
001: img1
002: img2
003: imga1
004: imga2
005: imga5
006: imga6
007: imga7
008: imgb0
009: imgb1
010: imgb2
011: imgb3
012: imgb4
013: imgb5
014: imgb6
015: imgb7
016: imgb8
017: imgb9
018: imgc1
019: imgc2
020: imgc4
021: imgc5
022: imgc7
023: imgc8
024: imgc9
025: imgd2
026: imgd3
027: imgd4
028: imgd7
029: imgd8
030: imgd9
031: imge0
032: imge1
033: imge2
034: imge3
035: imge4

---------Havard_test---------
001: imge7
002: imgf3
003: imgf4
004: imgf5
005: imgf7
006: imgh0
007: imgh1
008: imgh2
009: imgh3

For CAVE dataset, 30 scenes are used for training. These images are published in 
https://www1.cs.columbia.edu/CAVE/projects/gap_camera/. The filenames are 
as follows:
---------CAVE_train---------
001: balloons_ms
002: beads_ms
003: cd_ms
004: chart_and_stuffed_toy_ms
005: clay_ms
006: cloth_ms
007: egyptian_statue_ms
008: face_ms
009: beers_ms
010: lemon_slices_ms
011: lemons_ms
012: peppers_ms
013: strawberries_ms
014: sushi_ms
015: tomatoes_ms
016: feathers_ms
017: flowers_ms
018: glass_tiles_ms
019: hairs_ms
020: jelly_beans_ms
021: oil_painting_ms
022: paints_ms
023: photo_and_face_ms
024: pompoms_ms
025: yellowpeppers_ms
026: sponges_ms
027: stuffed_toys_ms
028: superballs_ms
029: thread_spools_ms
030: watercolors_ms

For KAIST dataset, 10 scenes are used for testing. These images are published in 
http://vclab.kaist.ac.kr/siggraphasia2017p1/. The filenames are as follows:
---------KAIST_test---------
001: scene03
002: scene04
003: scene05
004: scene08
005: scene11
006: scene12
007: scene14
008: scene21
009: scene23
010: scene24