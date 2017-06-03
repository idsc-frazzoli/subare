// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.LinkedList;
import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.tensor.Tensor;

public enum EpisodeKickoff {
  ;
  static Random random = new Random();

  // ---
  public static EpisodeInterface create(MonteCarloInterface monteCarloInterface, PolicyInterface policyInterface) {
    Tensor starts = monteCarloInterface.startStates();
    Tensor start = starts.get(random.nextInt(starts.length()));
    if (monteCarloInterface.isTerminal(start))
      throw new RuntimeException();
    return new MonteCarloEpisode(monteCarloInterface, policyInterface, start, new LinkedList<>());
  }
}
