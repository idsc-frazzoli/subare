// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;

abstract class AbstractExploringStarts {
  private final MonteCarloInterface monteCarloInterface;
  private int batchIndex = -1; // incremented from constructor
  private Policy policy; // must be private
  private ExploringStartsBatch exploringStartBatch;
  private int episodeIndex = 0;

  AbstractExploringStarts(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
  }

  final void nextBatch() {
    ++batchIndex; // holds subsequent batch id that won't change during the next episodes
    exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    policy = batchPolicy();
  }

  public final void nextEpisode() {
    EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
    // ---
    protected_nextEpisode(episodeInterface);
    // ---
    ++episodeIndex;
    if (!exploringStartBatch.hasNext())
      nextBatch();
  }

  public final int batchIndex() {
    return batchIndex;
  }

  public final int episodeIndex() {
    return episodeIndex;
  }

  public abstract void protected_nextEpisode(EpisodeInterface episodeInterface);

  public abstract Policy batchPolicy();
}
