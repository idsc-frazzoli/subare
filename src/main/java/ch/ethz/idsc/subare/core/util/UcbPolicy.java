// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions */
public class UcbPolicy extends PolicyBase {
  public UcbPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Tensor states) {
    super(discreteModel, qsa, sac, states);
  }

  @Override
  protected void appendToMaps(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().parallel(). //
        map(action -> UcbUtils.getUpperConfidenceBound(state, action, qsa.value(state, actions), sac, discreteModel)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    Tensor feasible = Tensor.of(fairArgMax.options().stream().map(actions::get));
    stateToBestActions.put(state, Index.build(feasible));
    stateToActionSize.put(state, actions.length());
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    Index index = stateToBestActions.get(state);
    final int optimalCount = index.size();
    return index.containsKey(action) ? RationalScalar.of(1, optimalCount) : RealScalar.ZERO;
  }
}
