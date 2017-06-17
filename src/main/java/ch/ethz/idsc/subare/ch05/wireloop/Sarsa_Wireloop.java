// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Round;

class Sarsa_Wireloop {
  static void handle(SarsaType sarsaType, int nstep, int EPISODES) throws Exception {
    System.out.println(sarsaType);
    String name = "wire5";
    Tensor grad = Tensors.vector(-1.1, .5);
    Clip clip = Clip.function(-1, .5);
    Function<Tensor, Scalar> stepCost = action -> clip.apply(action.dot(grad).Get());
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x, stepCost);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    Tensor epsilon = Subdivide.of(.2, .01, EPISODES);
    // epsilon = epsilon.pmul(epsilon);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    System.out.println(qsa.size());
    Sarsa sarsa = sarsaType.supply(wireloop, qsa, DefaultLearningRate.of(3, 0.51));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "L_qsa_" + sarsaType + "" + nstep + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar loss = Loss.accumulation(wireloop, ref, qsa);
      System.out.println(index + " " + epsilon.Get(index).map(Round._2) + " " + loss.map(Round._2));
      Policy policy = EGreedyPolicy.bestEquiprobable(wireloop, qsa, epsilon.Get(index));
      sarsa.setPolicy(policy);
      ExploringStarts.batch(wireloop, policy, nstep, sarsa);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, ref, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.expected, 1, 20);
  }
}
