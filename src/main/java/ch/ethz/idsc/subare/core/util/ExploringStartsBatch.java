// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;

/* apckage */ class ExploringStartsBatch {
  private final MonteCarloInterface monteCarloInterface;
  private final List<Tensor> list;
  private int index = 0;

  /* package */ ExploringStartsBatch(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
    Index index = DiscreteUtils.build(monteCarloInterface, monteCarloInterface.startStates());
    list = index.keys().flatten(0).collect(Collectors.toList());
    Collections.shuffle(list);
  }

  public boolean hasNext() {
    return index < list.size();
  }

  public EpisodeInterface nextEpisode(PolicyInterface policyInterface) {
    Tensor key = list.get(index);
    Tensor state = key.get(0); // first state
    Tensor action = key.get(1); // first action
    if (monteCarloInterface.isTerminal(state)) // consistency check
      throw new RuntimeException();
    Queue<Tensor> queue = new LinkedList<>();
    queue.add(action);
    ++index;
    return new MonteCarloEpisode(monteCarloInterface, policyInterface, state, queue);
  }
}
