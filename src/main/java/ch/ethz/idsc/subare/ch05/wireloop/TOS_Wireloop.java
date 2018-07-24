// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.LearningRate;
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
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;

enum TOS_Wireloop {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.3);
  private static final Scalar EPSILON = RealScalar.of(0.1);

  static void run(SarsaType sarsaType) throws Exception {
    String name = "wire4";
    System.out.println(sarsaType);
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    // Gambler gambler = new Gambler(20, RealScalar.of(.4));
    final DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    FeatureMapper mapper = ExactFeatureMapper.of(wireloop);
    FeatureWeight w = new FeatureWeight(mapper);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(wireloop, LAMBDA, mapper, learningRate, w);
    trueOnlineSarsa.setExplore(EPSILON);
    final String algo = sarsaType.name().toLowerCase();
    Stopwatch stopwatch = Stopwatch.started();
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures(name + "_tos_" + algo + ".gif"), 250)) {
      for (int batch = 0; batch < 20; ++batch) {
        // System.out.println("batch " + batch);
        Policy policy = EGreedyPolicy.bestEquiprobable(wireloop, trueOnlineSarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(wireloop, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsa = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(wireloop, batch, ref, qsa);
        gsw.append(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
        if (infoline.isLossfree()) {
          gsw.append(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
          gsw.append(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
          gsw.append(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
          break;
        }
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + stopwatch.display_seconds() + "s");
  }

  public static void main(String[] args) throws Exception {
    for (SarsaType sarsaType : SarsaType.values())
      run(sarsaType);
  }
}
