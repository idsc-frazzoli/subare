// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** p.33 */
public class EGreedyPolicy extends PolicyBase {
  protected ExplorationRate explorationRate;

  /* package */ EGreedyPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    super(discreteModel, qsa, sac);
    explorationRate = ConstantExplorationRate.of(0.1);
  }

  /* package */ EGreedyPolicy(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    super(standardModel, vs, sac);
    explorationRate = ConstantExplorationRate.of(0.1);
  }

  public EGreedyPolicy setExplorationRate(ExplorationRate explorationRate) {
    this.explorationRate = explorationRate;
    return this;
  }

  @Override
  protected Tensor getBestActions(DiscreteModel discreteModel, Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().map(action -> qsa.value(state, action)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    return Tensor.of(fairArgMax.options().stream().map(actions::get));
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    Tensor bestActions = getBestActions(discreteModel, state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    final int nonOptimalCount = discreteModel.actions(state).length() - optimalCount;
    if (nonOptimalCount == 0) // no non-optimal action exists
      return RationalScalar.of(1, optimalCount);
    Scalar epsilon = explorationRate.epsilon(state, sac);
    if (index.containsKey(action))
      return RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount));
    return epsilon.divide(RealScalar.of(nonOptimalCount));
  }

  @Override
  public PolicyBase copyOf(PolicyBase policyBase) {
    GlobalAssert.that(policyBase instanceof EGreedyPolicy);
    EGreedyPolicy newPolicy = new EGreedyPolicy(policyBase.discreteModel, policyBase.qsa, policyBase.sac);
    return newPolicy.setExplorationRate(((EGreedyPolicy) policyBase).explorationRate);
  }
}
