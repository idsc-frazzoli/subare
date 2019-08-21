// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Timing;

enum TOS_Gridworld {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.5);

  static void run(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    FeatureMapper mapper = ExactFeatureMapper.of(gridworld);
    FeatureWeight w = new FeatureWeight(mapper);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(gridworld, DiscreteQsa.build(gridworld), sac);
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gridworld, LAMBDA, mapper, learningRate, w, sac, policy);
    final String name = sarsaType.name().toLowerCase();
    Timing timing = Timing.started();
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gridworld_tos_" + name + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int batch = 0; batch < 100; ++batch) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        policy.setQsa(trueOnlineSarsa.qsaInterface());
        ExploringStarts.batch(gridworld, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsa = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(gridworld, batch, ref, qsa);
        animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        if (infoline.isLossfree()) {
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          break;
        }
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + timing.seconds() + "s");
  }

  public static void main(String[] args) throws Exception {
    run(SarsaType.ORIGINAL);
    run(SarsaType.EXPECTED);
    run(SarsaType.QLEARNING);
  }
}
