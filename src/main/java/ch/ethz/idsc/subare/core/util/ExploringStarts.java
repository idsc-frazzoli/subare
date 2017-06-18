// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;

/** contains helper functions to launch batches of episodes
 * that satisfy the exploring starts condition and have them processed by
 * {@link StepDigest}, {@link DequeDigest}, or {@link EpisodeDigest} */
public enum ExploringStarts {
  ;
  public static int batch( //
      MonteCarloInterface monteCarloInterface, Policy policy, //
      EpisodeDigest episodeDigest) {
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
      episodeDigest.digest(episodeInterface);
      ++episodes;
    }
    return episodes;
  }

  public static int batchWithReplay( //
      MonteCarloInterface monteCarloInterface, Policy policy, //
      EpisodeDigest... episodeDigest) {
    List<EpisodeDigest> list = Arrays.asList(episodeDigest);
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
      EpisodeRecording episodeRecording = new EpisodeRecording(episodeInterface);
      list.stream().parallel() //
          .forEach(_episodeDigest -> _episodeDigest.digest(episodeRecording.replay()));
      ++episodes;
    }
    return episodes;
  }

  public static int batch( //
      MonteCarloInterface monteCarloInterface, Policy policy, //
      StepDigest... stepDigest) {
    List<StepDigest> list = Arrays.asList(stepDigest);
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
      while (episodeInterface.hasNext()) {
        StepInterface stepInterface = episodeInterface.step();
        list.stream().parallel() //
            .forEach(_stepDigest -> _stepDigest.digest(stepInterface));
      }
      ++episodes;
    }
    return episodes;
  }

  /** @param monteCarloInterface
   * @param policy
   * @param nstep of deque (if nstep == 0 then deque contains a complete episode)
   * @param dequeDigest
   * @return */
  public static int batch( //
      MonteCarloInterface monteCarloInterface, Policy policy, int nstep, //
      DequeDigest... dequeDigest) {
    List<DequeDigest> list = Arrays.asList(dequeDigest);
    ExploringStartsBatch exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    int episodes = 0;
    while (exploringStartBatch.hasNext()) {
      EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
      Deque<StepInterface> deque = new LinkedList<>();
      while (episodeInterface.hasNext()) {
        final StepInterface stepInterface = episodeInterface.step();
        deque.add(stepInterface);
        if (deque.size() == nstep) {
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
      ++episodes;
    }
    return episodes;
  }
}
