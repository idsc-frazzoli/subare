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
  List<Agent> list;

  Judger(Bandits bandit, Agent... players) {
    this.bandit = bandit;
    list = Arrays.asList(players);
  }

  void play() {
    bandit.pullAll();
    for (Agent player : list) {
      int k = player.takeAction();
      RealScalar value = bandit.getLever(k);
      player.feedReward(k, value);
    }
  }

  void ranking() {
    Map<RealScalar, Agent> map = new TreeMap<>();
    list.stream().forEach(p -> map.put(p.getTotal(), p));
    for (Agent player : map.values()) {
      Scalar s = player.getTotal().minus(bandit.min) //
          .divide(bandit.max.minus(bandit.min)).multiply(RealScalar.of(100));
      System.out.println(player + "\t" + Round.of(s) + " %");
    }
  }
}
