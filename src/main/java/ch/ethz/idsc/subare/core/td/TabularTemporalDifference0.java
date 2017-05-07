// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;

/** for estimating value of policy
 * using eq (6.2) on p.128
 * 
 * V(S) = V(S) + alpha * [R + gamma * V(S') - V(S)]
 * 
 * see box on p.128 */
public class TabularTemporalDifference0 {
  private final StandardModel standardModel;
  private final PolicyInterface policyInterface;
  private final Scalar alpha;
  private final Scalar gamma;
  private final EpisodeSupplier episodeSupplier;

  public TabularTemporalDifference0( //
      StandardModel standardModel, PolicyInterface policyInterface, Scalar alpha, Scalar gamma, EpisodeSupplier episodeSupplier) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
    this.alpha = alpha;
    this.gamma = gamma;
    this.episodeSupplier = episodeSupplier;
  }

  public Tensor simulate(final int iterations) {
    int iteration = 0;
    Tensor states = standardModel.states();
    Index statesIndex = Index.build(states);
    Tensor values = Array.zeros(statesIndex.size());
    while (iteration < iterations) {
      EpisodeInterface episodeInterface = episodeSupplier.kickoff(policyInterface);
      while (episodeInterface.hasNext()) {
        StepInterface stepInterface = episodeInterface.step();
        Tensor state = stepInterface.prevState();
        Scalar reward = stepInterface.reward();
        Tensor stateS = stepInterface.nextState();
        int anteI = statesIndex.of(state);
        int nextI = statesIndex.of(stateS);
        Scalar delta = reward.add(values.get(nextI).multiply(gamma)).subtract(values.get(anteI)).multiply(alpha);
        values.set(scalar -> scalar.add(delta), anteI);
      }
      ++iteration;
    }
    return values;
  }
}
