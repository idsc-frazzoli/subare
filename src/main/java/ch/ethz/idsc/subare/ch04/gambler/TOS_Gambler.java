// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Timing;

/* package */ enum TOS_Gambler {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  static void run(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    Gambler gambler = new Gambler(20, RealScalar.of(.4));
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    FeatureMapper mapper = ExactFeatureMapper.of(gambler);
    FeatureWeight w = new FeatureWeight(mapper);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, qsa, sac);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gambler, LAMBDA, mapper, learningRate, w, sac, policy);
    final String name = sarsaType.name().toLowerCase();
    Timing timing = Timing.started();
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_tos_" + name + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int batch = 0; batch < 100; ++batch) {
        // System.out.println("batch " + batch);
        policy.setQsa(trueOnlineSarsa.qsaInterface());
        ExploringStarts.batch(gambler, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsaRef = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(gambler, batch, ref, qsaRef);
        animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gambler), qsaRef, ref));
        if (infoline.isLossfree()) {
          animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gambler), qsaRef, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gambler), qsaRef, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gambler), qsaRef, ref));
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
