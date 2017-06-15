// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

/** Double Sarsa for gridworld */
class Double_Gridworld {
  static void handle(SarsaType sarsaType, int n) throws Exception {
    System.out.println("double " + sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int EPISODES = 40;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES); // used in egreedy
    DiscreteQsa qsa1 = DiscreteQsa.build(gridworld);
    DiscreteQsa qsa2 = DiscreteQsa.build(gridworld);
    DoubleSarsa doubleSarsa = new DoubleSarsa(sarsaType, gridworld, //
        qsa1, qsa2, //
        DefaultLearningRate.of(5, .51), //
        DefaultLearningRate.of(5, .51));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gridworld_double_" + sarsaType + "" + n + ".gif"), 150);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar explore = epsilon.Get(index);
      Scalar error = TensorValuesUtils.distance(qsa1, ref);
      System.out.println(index + " " + explore.map(Round._2) + " " + error.map(Round._1));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable( //
          gridworld, TensorValuesUtils.average(qsa1, qsa2), explore);
      doubleSarsa.setPolicyInterface(policyInterface);
      ExploringStarts.batch(gridworld, policyInterface, n, doubleSarsa);
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, qsa1, ref)));
    }
    gsw.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa1);
    Put.of(UserHome.file("gridworld_" + sarsaType), vs.values());
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(gridworld, qsa1);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyInterface);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 1);
    // handle(SarsaType.expected, 3);
    handle(SarsaType.qlearning, 1);
  }
}
