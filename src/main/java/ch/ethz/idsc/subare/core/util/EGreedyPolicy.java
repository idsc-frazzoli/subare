// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Clip;

/** p.33 */
public class EGreedyPolicy extends PolicyBase {
  /** probability of choosing a non-optimal action, if there is at least one non-optimal action */
  private final Scalar epsilon;

  public EGreedyPolicy(DiscreteModel discreteModel, QsaInterface qsa, Scalar epsilon, Tensor states) {
    super(discreteModel, qsa, null, states);
    Clip.function(0, 1).requireInside(epsilon);
    this.epsilon = epsilon;
  }

  public EGreedyPolicy(DiscreteModel discreteModel, QsaInterface qsa, Scalar epsilon) {
    super(discreteModel, qsa, null, discreteModel.states());
    Clip.function(0, 1).requireInside(epsilon);
    this.epsilon = epsilon;
  }

  public EGreedyPolicy(StandardModel standardModel, VsInterface vs, Scalar epsilon, Tensor states) {
    states.forEach(v -> appendToMaps(standardModel, vs, v));
    Clip.function(0, 1).requireInside(epsilon);
    this.epsilon = epsilon;
  }

  @Override
  protected void appendToMaps(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().map(action -> qsa.value(state, action)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    Tensor feasible = Tensor.of(fairArgMax.options().stream().map(actions::get));
    stateToBestActions.put(state, Index.build(feasible));
    stateToActionSize.put(state, actions.length());
  }

  protected void appendToMaps(StandardModel standardModel, VsInterface vs, Tensor state) {
    ActionValueAdapter actionValueAdapter = new ActionValueAdapter(standardModel);
    Tensor actions = standardModel.actions(state);
    Tensor va = Tensor.of(actions.stream() //
        .map(action -> actionValueAdapter.qsa(state, action, vs)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    Tensor feasible = Tensor.of(fairArgMax.options().stream().map(actions::get));
    stateToBestActions.put(state, Index.build(feasible));
    stateToActionSize.put(state, actions.length());
  }

  @Override // from Policy
  public Scalar probability(Tensor state, Tensor action) {
    Index index = stateToBestActions.get(state);
    final int optimalCount = index.size();
    final int nonOptimalCount = stateToActionSize.get(state) - optimalCount;
    if (nonOptimalCount == 0) // no non-optimal action exists
      return RationalScalar.of(1, optimalCount);
    if (index.containsKey(action))
      return RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount));
    return epsilon.divide(RealScalar.of(nonOptimalCount));
  }
}
