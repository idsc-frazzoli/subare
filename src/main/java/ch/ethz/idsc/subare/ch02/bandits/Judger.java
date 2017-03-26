// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Round;

class Judger {
  final Bandits bandit;
  final List<Agent> list;

  Judger(Bandits bandit, Agent... agents) {
    this.bandit = bandit;
    list = Arrays.asList(agents);
  }

  void play() {
    bandit.pullAll();
    for (Agent agent : list) {
      int k = agent.takeAction();
      Scalar value = bandit.getLever(k);
      agent.feedback(k, value);
    }
  }

  void ranking() {
    Map<Scalar, Agent> map = new TreeMap<>();
    list.stream().forEach(p -> map.put(p.getRewardTotal(), p));
    for (Agent agent : map.values()) {
      Scalar s = agent.getRewardTotal().subtract(bandit.min) //
          .divide(bandit.max.subtract(bandit.min)).multiply(RealScalar.of(100));
      System.out.println(String.format("%25s%5s %%%8s RND", //
          agent, Round.of(s), "" + agent.getRandomizedDecisionCount()));
    }
  }
}
