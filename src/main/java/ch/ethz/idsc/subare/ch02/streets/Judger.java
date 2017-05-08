// code by jph
package ch.ethz.idsc.subare.ch02.streets;

import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.FairMaxAgent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;

class Judger {
  final Network network;
  final List<Agent> list;

  Judger(Network network, Agent... agents) {
    this.network = network;
    list = Arrays.asList(agents);
  }

  void play() {
    network.reset();
    // agents choose path
    for (Agent agent : list)
      network.feedAction(agent.takeAction());
    Tensor cost = network.cost();
    // network computes costs
    for (Agent agent : list) {
      int k = agent.getActionReminder();
      Scalar total = ZeroScalar.get();
      for (int s : network.streetsFromAction(k))
        total = total.add(cost.Get(s));
      agent.feedback(k, total);
    }
  }

  public static void main(String[] args) {
    FairMaxAgent a1 = new OptimistAgent(3, RealScalar.of(5), RealScalar.of(.1));
    FairMaxAgent a2 = new OptimistAgent(3, RealScalar.of(5), RealScalar.of(.1));
    FairMaxAgent a3 = new OptimistAgent(3, RealScalar.of(5), RealScalar.of(.1));
    a1.setOpeningSequence(0);
    a2.setOpeningSequence(1);
    a3.setOpeningSequence(2);
    Agent[] agents = new Agent[] { a1, a2, a3 };
    Judger judger = new Judger(new BridgeNetwork(), agents);
    for (int rnd = 0; rnd < 1000; ++rnd)
      judger.play();
    for (Agent a : agents)
      System.out.println(a.getRewardAverage());
    // System.out.println(a2.getRewardTotal());
    // System.out.println(a3.getRewardTotal());
  }
}
