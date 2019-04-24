// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Round;

/* package */ class Judger {
  private final Bandits bandit;
  private final Map<Agent, Tensor> map = new HashMap<>();

  Judger(Bandits bandit, Agent... agents) {
    this.bandit = bandit;
    Stream.of(agents).forEach(agent -> map.put(agent, Tensors.empty()));
  }

  void play() {
    Tensor tensor = bandit.pullAll();
    for (Agent agent : map.keySet()) {
      int k = agent.takeAction();
      Scalar value = tensor.Get(k);
      agent.feedback(k, value);
      map.get(agent).append(agent.getRewardTotal());
    }
  }

  void ranking() {
    List<Agent> list = new ArrayList<>(map.keySet());
    Collections.sort(list, (a1, a2) -> Scalars.compare(a1.getRewardTotal(), a2.getRewardTotal()));
    Clip clip = bandit.clip();
    for (Agent agent : list) {
      Scalar s = clip.rescale(agent.getRewardTotal()).multiply(RealScalar.of(100));
      System.out.println(String.format("%25s%5s %%%8s RND", //
          agent, Round.of(s), "" + agent.getRandomizedDecisionCount()));
    }
  }

  Map<Agent, Tensor> map() {
    return Collections.unmodifiableMap(map);
  }
}
