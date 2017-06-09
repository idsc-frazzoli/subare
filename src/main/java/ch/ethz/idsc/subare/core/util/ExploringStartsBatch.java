// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Collections;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;

public class ExploringStartsBatch {
  public static int apply(MonteCarloInterface monteCarloInterface, PolicyInterface policyInterface, EpisodeDigest episodeDigest) {
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policyInterface);
      episodeDigest.digest(episodeInterface);
      ++episodes;
    }
    return episodes;
  }

  // TODO ordering of arguments not intuitive
  public static int apply(MonteCarloInterface monteCarloInterface, StepDigest stepDigest, PolicyInterface policyInterface) {
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policyInterface);
      while (episodeInterface.hasNext())
        stepDigest.digest(episodeInterface.step());
      ++episodes;
    }
    return episodes;
  }

  // TODO ordering of arguments not intuitive
  public static int apply(MonteCarloInterface monteCarloInterface, DequeDigest stepDigest, int size, PolicyInterface policyInterface) {
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policyInterface);
      Deque<StepInterface> queue = new LinkedList<>();
      while (episodeInterface.hasNext()) {
        queue.add(episodeInterface.step());
        if (queue.size() == size) {
          stepDigest.digest(queue);
          queue.poll();
        }
      }
      while (!queue.isEmpty()) {
        stepDigest.digest(queue);
        queue.poll();
      }
      ++episodes;
    }
    return episodes;
  }

  // BlockDigest
  public static ExploringStartsBatch create(MonteCarloInterface monteCarloInterface) {
    return new ExploringStartsBatch(monteCarloInterface);
  }

  private final MonteCarloInterface monteCarloInterface;
  private final List<Tensor> list;
  private int index = 0;

  private ExploringStartsBatch(MonteCarloInterface monteCarloInterface) {
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
    Tensor start = key.get(0);
    if (monteCarloInterface.isTerminal(start)) // consistency check
      throw new RuntimeException();
    Tensor action = key.get(1);
    Queue<Tensor> queue = new LinkedList<>();
    queue.add(action);
    ++index;
    return new MonteCarloEpisode(monteCarloInterface, policyInterface, start, queue);
  }
}
