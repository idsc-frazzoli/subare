// code by jph
package ch.ethz.idsc.subare.ch02.streets;

import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.Scalar;

public class Judger {
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
    // network computes costs
    for (Agent agent : list) {
      int k = agent.getActionReminder();
      Scalar value = network.costOfAction(k);
      agent.feedback(k, value);
    }
  }
}
