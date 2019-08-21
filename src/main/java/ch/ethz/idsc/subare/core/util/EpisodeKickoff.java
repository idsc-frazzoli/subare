// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.ArrayDeque;
import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.tensor.Tensor;

public enum EpisodeKickoff {
  ;
  private static final Random RANDOM = new Random();

  public static EpisodeInterface single(MonteCarloInterface monteCarloInterface, Policy policy) {
    Tensor starts = monteCarloInterface.startStates();
    Tensor start = starts.get(RANDOM.nextInt(starts.length()));
    return single(monteCarloInterface, policy, start);
  }

  public static EpisodeInterface single(MonteCarloInterface monteCarloInterface, Policy policy, Tensor start) {
    if (monteCarloInterface.isTerminal(start))
      throw new IllegalStateException();
    return new MonteCarloEpisode(monteCarloInterface, policy, start, new ArrayDeque<>());
  }
}
