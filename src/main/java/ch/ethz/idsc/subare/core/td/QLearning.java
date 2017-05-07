// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteModels;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;

/** Q-learning: An off-policy TD control algorithm
 * 
 * box on p.140 */
public class QLearning {
  private final StandardModel standardModel;
  private final PolicyInterface policyInterface;
  final PolicyWrap policyWrap;
  private final Scalar alpha;
  private final Scalar gamma;
  private final EpisodeSupplier episodeSupplier;
  private final Index qsa;
  final Tensor values;

  // TODO not final location: make Qsa an interface and DiscreteQsa an implementation
  private static Tensor lookup(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  public QLearning( //
      StandardModel standardModel, PolicyInterface policyInterface, //
      Scalar alpha, Scalar gamma, EpisodeSupplier episodeSupplier) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
    policyWrap = new PolicyWrap(policyInterface);
    this.alpha = alpha;
    this.gamma = gamma;
    this.episodeSupplier = episodeSupplier;
    qsa = DiscreteModels.build(standardModel);
    values = Array.zeros(qsa.size());
  }

  public Tensor simulate(final int iterations) {
    int iteration = 0;
    while (iteration < iterations) {
      step();
      ++iteration;
    }
    return values;
  }

  private void step() {
    EpisodeInterface episodeInterface = episodeSupplier.kickoff(policyInterface);
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      Tensor state0 = stepInterface.prevState();
      Tensor action0 = stepInterface.action();
      Scalar reward = stepInterface.reward();
      Tensor state1 = stepInterface.nextState();
      Scalar max = standardModel.actions(state1).flatten(0) //
          .map(action1 -> values.Get(qsa.of(lookup(state1, action1)))) //
          .reduce(Max::of).get();
      int anteI = qsa.of(lookup(state0, action0));
      Scalar delta = reward.add(max.multiply(gamma)).subtract(values.get(anteI)).multiply(alpha);
      values.set(scalar -> scalar.add(delta), anteI);
    }
  }

  public Index getQsa() {
    return qsa;
  }
}
