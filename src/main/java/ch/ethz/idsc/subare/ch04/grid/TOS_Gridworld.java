// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;

enum TOS_Gridworld {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.5);
  private static final Scalar EPSILON = RealScalar.of(0.1);

  static void run(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    FeatureMapper mapper = ExactFeatureMapper.of(gridworld);
    FeatureWeight w = new FeatureWeight(mapper);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gridworld, LAMBDA, mapper, learningRate, w);
    trueOnlineSarsa.setExplore(EPSILON);
    final String name = sarsaType.name().toLowerCase();
    Stopwatch stopwatch = Stopwatch.started();
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_tos_" + name + ".gif"), 250)) {
      for (int batch = 0; batch < 100; ++batch) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policy = EGreedyPolicy.bestEquiprobable(gridworld, trueOnlineSarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(gridworld, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsa = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(gridworld, batch, ref, qsa);
        gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        if (infoline.isLossfree()) {
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          break;
        }
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + stopwatch.display_seconds() + "s");
  }

  public static void main(String[] args) throws Exception {
    run(SarsaType.ORIGINAL);
    run(SarsaType.EXPECTED);
    run(SarsaType.QLEARNING);
  }
}
