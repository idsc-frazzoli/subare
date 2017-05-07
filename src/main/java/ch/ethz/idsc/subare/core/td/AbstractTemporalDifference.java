// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;

public abstract class AbstractTemporalDifference implements StepDigest {
  private final EpisodeSupplier episodeSupplier;
  private final PolicyInterface policyInterface;

  public AbstractTemporalDifference(EpisodeSupplier episodeSupplier, PolicyInterface policyInterface) {
    this.episodeSupplier = episodeSupplier;
    this.policyInterface = policyInterface;
  }

  public final void simulate(final int iterations) {
    int iteration = 0;
    while (iteration < iterations) {
      EpisodeInterface episodeInterface = episodeSupplier.kickoff(policyInterface);
      while (episodeInterface.hasNext())
        digest(episodeInterface.step());
      ++iteration;
    }
  }
}
