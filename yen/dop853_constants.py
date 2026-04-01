import numpy as np

N_STAGES = 13

A = np.array([
    [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0)],
    [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0)],
    [np.float64(0.05260015195876773), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0)],
    [np.float64(0.0197250569845379), np.float64(0.0591751709536137), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0)],
    [np.float64(0.02958758547680685), np.float64(0.0), np.float64(0.08876275643042054), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0)],
    [np.float64(0.2413651341592667), np.float64(0.0), np.float64(-0.8845494793282861), np.float64(0.924834003261792),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0)],
    [np.float64(0.037037037037037035), np.float64(0.0), np.float64(0.0), np.float64(0.17082860872947386),
     np.float64(0.12546768756682242), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)],
    [np.float64(0.037109375), np.float64(0.0), np.float64(0.0), np.float64(0.17025221101954405),
     np.float64(0.06021653898045596), np.float64(-0.017578125), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)],
    [np.float64(0.03709200011850479), np.float64(0.0), np.float64(0.0), np.float64(0.17038392571223998),
     np.float64(0.10726203044637328), np.float64(-0.015319437748624402), np.float64(0.008273789163814023),
     np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)],
    [np.float64(0.6241109587160757), np.float64(0.0), np.float64(0.0), np.float64(-3.3608926294469414),
     np.float64(-0.868219346841726), np.float64(27.59209969944671), np.float64(20.154067550477894),
     np.float64(-43.48988418106996), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
     np.float64(0.0)],
    [np.float64(0.47766253643826434), np.float64(0.0), np.float64(0.0), np.float64(-2.4881146199716677),
     np.float64(-0.590290826836843), np.float64(21.230051448181193), np.float64(15.279233632882423),
     np.float64(-33.28821096898486), np.float64(-0.020331201708508627), np.float64(0.0), np.float64(0.0),
     np.float64(0.0), np.float64(0.0)],
    [np.float64(-0.9371424300859873), np.float64(0.0), np.float64(0.0), np.float64(5.186372428844064),
     np.float64(1.0914373489967295), np.float64(-8.149787010746927), np.float64(-18.52006565999696),
     np.float64(22.739487099350505), np.float64(2.4936055526796523), np.float64(-3.0467644718982196), np.float64(0.0),
     np.float64(0.0), np.float64(0.0)],
    [np.float64(2.273310147516538), np.float64(0.0), np.float64(0.0), np.float64(-10.53449546673725),
     np.float64(-2.0008720582248625), np.float64(-17.9589318631188), np.float64(27.94888452941996),
     np.float64(-2.8589982771350235), np.float64(-8.87285693353063), np.float64(12.360567175794303),
     np.float64(0.6433927460157636), np.float64(0.0), np.float64(0.0)],
], dtype=np.float64)

B = np.array([np.float64(0.054293734116568765), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
              np.float64(4.450312892752409), np.float64(1.8915178993145003), np.float64(-5.801203960010585),
              np.float64(0.3111643669578199), np.float64(-0.1521609496625161), np.float64(0.20136540080403034),
              np.float64(0.04471061572777259), np.float64(0.0)], dtype=np.float64)

C = np.array([np.float64(0.0), np.float64(0.0), np.float64(0.05260015195876773), np.float64(0.0789002279381516),
              np.float64(0.1183503419072274), np.float64(0.2816496580927726), np.float64(0.3333333333333333),
              np.float64(0.25), np.float64(0.3076923076923077), np.float64(0.6512820512820513), np.float64(0.6),
              np.float64(0.8571428571428571), np.float64(1.0)], dtype=np.float64)

E3 = np.array([np.float64(-0.18980075407240762), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
               np.float64(4.450312892752409), np.float64(1.8915178993145003), np.float64(-5.801203960010585),
               np.float64(-0.4226823213237919), np.float64(-0.1521609496625161), np.float64(0.20136540080403034),
               np.float64(0.02265179219836082), np.float64(0.0)], dtype=np.float64)

E5 = np.array([np.float64(0.01312004499419488), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
               np.float64(-1.2251564463762044), np.float64(-0.4957589496572502), np.float64(1.6643771824549864),
               np.float64(-0.35032884874997366), np.float64(0.3341791187130175), np.float64(0.08192320648511571),
               np.float64(-0.022355307863886294), np.float64(0.0)], dtype=np.float64)
