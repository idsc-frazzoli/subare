// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.tensor.Tensor;

public class ExploringStartsBatch {
  public static void apply(MonteCarloInterface monteCarloInterface, EpisodeDigest episodeDigest, PolicyInterface policyInterface) {
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    while (exploringStartBatch.hasNext())
      episodeDigest.digest(exploringStartBatch.nextEpisode(policyInterface));
  }

  public static void apply(MonteCarloInterface monteCarloInterface, StepDigest stepDigest, PolicyInterface policyInterface) {
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policyInterface);
      while (episodeInterface.hasNext())
        stepDigest.digest(episodeInterface.step());
    }
  }

  public static ExploringStartsBatch create(MonteCarloInterface monteCarloInterface) {
    return new ExploringStartsBatch(monteCarloInterface);
  }

  private final MonteCarloInterface monteCarloInterface;
  private final List<Tensor> list;
  private int index = 0;
  private int actionIndex = 0;

  private ExploringStartsBatch(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
    list = monteCarloInterface.startStates().flatten(0).collect(Collectors.toList());
    Collections.shuffle(list);
  }

  public boolean hasNext() {
    return index < list.size();
  }

  public EpisodeInterface nextEpisode(PolicyInterface policyInterface) {
    Tensor start = list.get(index);
    if (monteCarloInterface.isTerminal(start)) // check
      throw new RuntimeException();
    Tensor actions = monteCarloInterface.actions(start);
    Tensor action = actions.get(actionIndex);
    ++actionIndex;
    if (actionIndex == actions.length()) {
      ++index;
      actionIndex = 0;
    }
    Queue<Tensor> queue = new LinkedList<>();
    queue.add(action);
    return new MonteCarloEpisode(monteCarloInterface, policyInterface, start, queue);
  }
}
