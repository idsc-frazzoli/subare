// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.Digits;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** Sarsa applied to gambler */
class Sarsa_Gambler {
  static Tensor train(Gambler gambler, SarsaType sarsaType, //
      int EPISODES, Scalar factor, Scalar exponent) throws Exception {
    System.out.println(sarsaType);
    final Tensor errors = Tensors.empty();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    Tensor epsilon = Subdivide.of(.02, .01, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(gambler); // q-function for training, initialized to 0
    // ---
    StateActionCounter sac = new StateActionCounter(gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gambler_qsa_" + sarsaType + ".gif"), 200);
    GifSequenceWriter gsc = GifSequenceWriter.of(UserHome.Pictures("gambler_sac_" + sarsaType + ".gif"), 200);
    // ---
    final Sarsa sarsa = sarsaType.supply(gambler, qsa, DefaultLearningRate.of(factor, exponent));
    // ---
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      errors.append(error);
      System.out.println(index + " " + epsilon.Get(index).map(Digits._1) + " " + error.map(Digits._1));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      sarsa.setPolicyInterface(policyInterface);
      sarsa.getUcbPolicy().setTime(RealScalar.of(index + 1)); // TODO
      PolicyInterface ucbPolicy = sarsa.getUcbPolicy();
      ExploringStarts.batch(gambler, //
          policyInterface //
          // ucbPolicy //
          , 1, sarsa, sac);
      // ---
      gsw.append(ImageFormat.of(GamblerHelper.qsaPolicyRef(gambler, qsa, ref)));
      gsc.append(ImageFormat.of(GamblerHelper.counts( //
          gambler, sac.qsa(StateActionCounter.LOGARITHMIC))));
    }
    gsw.close();
    gsc.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(gambler, qsa);
    EpisodeInterface mce = EpisodeKickoff.single(gambler, policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
    return errors;
  }

  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    // train(gambler, SarsaType.original, 100, 1.3, 0.51);
    Tensor errors = train(gambler, SarsaType.expected, 3, RealScalar.of(1.3), RealScalar.of(0.51));
    System.out.println(errors.map(Digits._1));
    // train(gambler, SarsaType.qlearning, 100, 0.2, 0.55);
    // train(gambler, SarsaType.qlearning, 100, 0.2, 0.55);
  }
}
