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

// TODO EXPERIMENTAL API not finalized
public abstract class ExploringStartsStream {
  private final MonteCarloInterface monteCarloInterface;
  private final int nstep;
  private final List<DequeDigest> list;
  private ExploringStartsBatch exploringStartBatch;
  private int batchIndex = 0;

  public ExploringStartsStream( //
      MonteCarloInterface monteCarloInterface, int nstep, DequeDigest... dequeDigest) {
    this.monteCarloInterface = monteCarloInterface;
    this.nstep = nstep;
    list = Arrays.asList(dequeDigest);
    exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
  }

  public boolean hasNextEpisode() {
    // TODO possibly use function in application in while loop
    return false;
  }

  public void nextEpisode() {
    if (!exploringStartBatch.hasNext()) {
      exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
      ++batchIndex;
    }
    // ---
    EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(providePolicy());
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
    // ++episodes;
  }

  public abstract Policy providePolicy();

  public int batchIndex() {
    return batchIndex;
  }
}
