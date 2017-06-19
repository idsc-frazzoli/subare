// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;

public abstract class ExploringStartsStream {
  private final MonteCarloInterface monteCarloInterface;
  private final int nstep;
  private final List<DequeDigest> list;
  private int batchIndex = -1; // incremented from constructor
  private ExploringStartsBatch exploringStartBatch;
  private Policy policy;
  private int episodeIndex = 0;

  public ExploringStartsStream( //
      MonteCarloInterface monteCarloInterface, int nstep, DequeDigest... dequeDigest) {
    this.monteCarloInterface = monteCarloInterface;
    this.nstep = nstep;
    list = Arrays.asList(dequeDigest);
  }

  public void nextEpisode() {
    if (exploringStartBatch == null || !exploringStartBatch.hasNext()) {
      ++batchIndex; // holds subsequent batch id that won't change during the next episodes
      exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
      policy = batchPolicy();
    }
    // ---
    EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
    Deque<StepInterface> deque = new LinkedList<>();
    while (episodeInterface.hasNext()) {
      final StepInterface stepInterface = episodeInterface.step();
      deque.add(stepInterface);
      if (deque.size() == nstep) { // never true, if nstep == 0
        list.stream().parallel() //
            .forEach(_dequeDigest -> _dequeDigest.digest(deque));
        deque.poll();
      }
    }
    while (!deque.isEmpty()) {
      list.stream().parallel() //
          .forEach(_dequeDigest -> _dequeDigest.digest(deque));
      deque.poll();
    }
    ++episodeIndex;
  }

  public abstract Policy batchPolicy();

  public int batchIndex() {
    return batchIndex;
  }

  public int episodeIndex() {
    return episodeIndex;
  }
}
