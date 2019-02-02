// code by jph
package ch.ethz.idsc.subare.demo.prison;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

class Judger {
  final Tensor reward;
  final Agent a1;
  final Agent a2;

  Judger(Tensor r1, Agent a1, Agent a2) {
    this.reward = r1;
    this.a1 = a1;
    this.a2 = a2;
  }

  void play() {
    int A1 = a1.takeAction();
    int A2 = a2.takeAction();
    a1.feedback(A1, reward.Get(A1, A2));
    a2.feedback(A2, reward.Get(A2, A1));
  }

  /** @return tensor of rewards averaged over number of actions */
  Tensor ranking() {
    GlobalAssert.that(a1.getCount().equals(a2.getCount()));
    return Tensors.of(a1.getRewardTotal(), a2.getRewardTotal()).divide(a1.getCount());
  }
}
