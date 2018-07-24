// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** Sarsa applied to gambler */
enum Sarsa_Gambler {
  ;
  static void train(Gambler gambler, SarsaType sarsaType, //
      int batches, Scalar factor, Scalar exponent) throws Exception {
    System.out.println(sarsaType);
    GamblerRaster gamblerRaster = new GamblerRaster(gambler);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    Tensor epsilon = Subdivide.of(.2, .01, batches);
    DiscreteQsa qsa = DiscreteQsa.build(gambler); // q-function for training, initialized to 0
    // ---
    StateActionCounter sac = new StateActionCounter(gambler);
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gambler_qsa_" + sarsaType + ".gif"), 150);
    AnimationWriter gsc = AnimationWriter.of(UserHome.Pictures("gambler_sac_" + sarsaType + ".gif"), 150);
    // ---
    final Sarsa sarsa = sarsaType.supply(gambler, DefaultLearningRate.of(factor, exponent), qsa);
    // ---
    for (int index = 0; index < batches; ++index) {
      Infoline.print(gambler, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      // sarsa.supplyPolicy(() -> policy);
      sarsa.setExplore(epsilon.Get(index));
      ExploringStarts.batch(gambler, policy, 1, sarsa, sac);
      // ---
      gsw.append(StateActionRasters.qsaPolicyRef(gamblerRaster, qsa, ref));
      gsc.append(StateActionRasters.qsa( //
          gamblerRaster, DiscreteValueFunctions.rescaled(sac.qsa(StateActionCounter.LOGARITHMIC))));
    }
    gsw.close();
    gsc.close();
    GamblerHelper.play(gambler, qsa);
  }

  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    gambler = new Gambler(20, RationalScalar.of(4, 10));
    train(gambler, SarsaType.QLEARNING, 20, RealScalar.of(3), RealScalar.of(0.81));
  }
}
