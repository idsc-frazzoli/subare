// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

/** Sarsa applied to gambler */
/* package */ class Sarsa_Gambler {
  private final GamblerModel gamblerModel;
  /** true q-function, for error measurement */
  private final DiscreteQsa ref;

  public Sarsa_Gambler(GamblerModel gamblerModel) {
    this.gamblerModel = gamblerModel;
    ref = GamblerHelper.getOptimalQsa(gamblerModel);
  }

  DiscreteQsa train(SarsaType sarsaType, int batches, LearningRate learningRate) throws Exception {
    System.out.println(sarsaType);
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel); // q-function for training, initialized to 0
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    // ---
    try (AnimationWriter animationWriter1 = new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_" + sarsaType + ".gif"), 150, TimeUnit.MILLISECONDS)) {
      try (AnimationWriter animationWriter2 = new GifAnimationWriter(HomeDirectory.Pictures("gambler_sac_" + sarsaType + ".gif"), 150, TimeUnit.MILLISECONDS)) {
        Sarsa sarsa = sarsaType.sarsa(gamblerModel, learningRate, qsa, sac, policy);
        for (int index = 0; index < batches; ++index) {
          Infoline.print(gamblerModel, index, ref, qsa);
          ExploringStarts.batch(gamblerModel, policy, 1, sarsa);
          // ---
          animationWriter1.write(StateActionRasters.qsaPolicyRef(gamblerRaster, qsa, ref));
          animationWriter2.write(StateActionRasters.qsa( //
              gamblerRaster, DiscreteValueFunctions.rescaled(((DiscreteStateActionCounter) sarsa.sac()).inQsa(gamblerModel))));
        }
      }
    }
    GamblerHelper.play(gamblerModel, qsa);
    return qsa;
  }

  public static void main(String[] args) throws Exception {
    GamblerModel gambler = new GamblerModel(20, RationalScalar.of(4, 10));
    Sarsa_Gambler sarsa_Gambler = new Sarsa_Gambler(gambler);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    DiscreteQsa qsa = sarsa_Gambler.train(SarsaType.QLEARNING, 20, learningRate);
  }
}
