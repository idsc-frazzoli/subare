// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.subare.core.util.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** Sarsa applied to gambler */
class Sarsa_Gambler {
  static void train(Gambler gambler, SarsaType sarsaType, //
      int batches, Scalar factor, Scalar exponent) throws Exception {
    System.out.println(sarsaType);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    Tensor epsilon = Subdivide.of(.2, .01, batches);
    DiscreteQsa qsa = DiscreteQsa.build(gambler); // q-function for training, initialized to 0
    // ---
    StateActionCounter sac = new StateActionCounter(gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gambler_qsa_" + sarsaType + ".gif"), 150);
    GifSequenceWriter gsc = GifSequenceWriter.of(UserHome.Pictures("gambler_sac_" + sarsaType + ".gif"), 150);
    // ---
    final Sarsa sarsa = sarsaType.supply(gambler, qsa, DefaultLearningRate.of(factor, exponent));
    // ---
    for (int index = 0; index < batches; ++index) {
      Infoline.print(gambler, index, ref, qsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      sarsa.supplyPolicy(() -> policy);
      ExploringStarts.batch(gambler, policy, 1, sarsa, sac);
      // ---
      gsw.append(ImageFormat.of(StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref)));
      gsc.append(ImageFormat.of(GamblerHelper.counts( //
          gambler, sac.qsa(StateActionCounter.LOGARITHMIC))));
    }
    gsw.close();
    gsc.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    Policy policy = GreedyPolicy.bestEquiprobable(gambler, qsa);
    EpisodeInterface mce = EpisodeKickoff.single(gambler, policy);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    gambler = new Gambler(100, RationalScalar.of(4, 10));
    train(gambler, SarsaType.qlearning, 20, RealScalar.of(3), RealScalar.of(0.51));
  }
}
