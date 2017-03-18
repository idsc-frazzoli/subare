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
      Scalar value = bandit.getLever(k);
      player.feedback(k, value);
    }
  }

  void ranking() {
    Map<Scalar, Agent> map = new TreeMap<>();
    list.stream().forEach(p -> map.put(p.getTotal(), p));
    for (Agent player : map.values()) {
      Scalar s = player.getTotal().subtract(bandit.min) //
          .divide(bandit.max.subtract(bandit.min)).multiply(RealScalar.of(100));
      System.out.println(player + "\t" + Round.of(s) + " %");
    }
  }
}
