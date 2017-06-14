// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.LearningRate;
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
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** Sarsa applied to gambler */
class Sarsa_Gambler {
  /** @param type
   * @param factor
   * @param exponent
   * @return errors
   * @throws Exception */
  static Tensor handle(Gambler gambler, SarsaType type, int EPISODES, double factor, double exponent) throws Exception {
    System.out.println(type);
    final Tensor errors = Tensors.empty();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    Tensor epsilon = Subdivide.of(.6, .01, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    StateActionCounter sac = new StateActionCounter(gambler);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gambler_qsa_" + type + ".gif"), 200);
    GifSequenceWriter gsc = GifSequenceWriter.of(UserHome.Pictures("gambler_sac_" + type + ".gif"), 200);
    LearningRate learningRate = DefaultLearningRate.of(factor, exponent);
    final Sarsa sarsa = type.supply(gambler, qsa, learningRate);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      errors.append(error);
      System.out.println(index + " " + epsilon.Get(index).map(Digits._1) + " " + error.map(Digits._1));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      sarsa.setPolicyInterface(policyInterface);
      for (int count = 0; count < 1; ++count)
        ExploringStarts.batch(gambler, policyInterface, 1, sarsa, sac);
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
    // handle(gambler, SarsaType.original, 100, 1.3, 0.51);
    // handle(gambler, SarsaType.expected, 100, 1.3, 0.51);
    handle(gambler, SarsaType.qlearning, 100, 0.2, 0.55);
  }
}
