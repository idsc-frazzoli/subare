// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;

/* apckage */ class ExploringStartsBatch {
  private final MonteCarloInterface monteCarloInterface;
  private final List<Tensor> list; /** contains all starting start-action pairs, shuffled randomly*/
  private int index = 0;

  /* package */ ExploringStartsBatch(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
    Index index = DiscreteUtils.build(monteCarloInterface, monteCarloInterface.startStates());
    list = index.keys().stream().collect(Collectors.toList());
    Collections.shuffle(list);
  }

  /** @return true if call to nextEpisode is valid */
  public boolean hasNext() {
    return index < list.size();
  }

  /** @param policy
   * @return
   * @throws Exception if hasNext() == false */
  public EpisodeInterface nextEpisode(Policy policy) {
    Tensor key = list.get(index);
    Tensor state = key.get(0); // first state
    Tensor action = key.get(1); // first action
    if (monteCarloInterface.isTerminal(state)) // consistency check
      throw new RuntimeException();
    Queue<Tensor> queue = new ArrayDeque<>();
    queue.add(action);
    ++index;
    return new MonteCarloEpisode(monteCarloInterface, policy, state, queue);
  }
}
