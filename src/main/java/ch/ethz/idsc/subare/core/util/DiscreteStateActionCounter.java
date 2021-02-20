// code by fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.api.ScalarUnaryOperator;
import ch.ethz.idsc.tensor.sca.Log;

public class DiscreteStateActionCounter implements StateActionCounter, Serializable {
  private static final ScalarUnaryOperator LOGARITHMIC = scalar -> Log.of(scalar.add(RealScalar.ONE));
  // ---
  private final Map<Tensor, Integer> stateActionMap = new HashMap<>();
  private final Map<Tensor, Integer> stateMap = new HashMap<>();

  @Override // from StepDigest
  public void digest(StepInterface stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    Tensor state = stepInterface.prevState();
    stateMap.merge(state, 1, Math::addExact);
    stateActionMap.merge(key, 1, Math::addExact);
  }

  @Override // from StateActionCounter
  public Scalar stateActionCount(Tensor key) {
    return stateActionMap.containsKey(key) //
        ? RealScalar.of(stateActionMap.get(key))
        : RealScalar.ZERO;
  }

  @Override // from StateActionCounter
  public Scalar stateCount(Tensor state) {
    return stateMap.containsKey(state) //
        ? RealScalar.of(stateMap.get(state))
        : RealScalar.ZERO;
  }

  @Override // from StateActionCounter
  public boolean isEncountered(Tensor key) {
    return stateActionMap.containsKey(key);
  }

  public void setStateCount(Tensor state, Scalar value) {
    stateMap.put(state, Scalars.intValueExact(value));
  }

  public void setStateActionCount(Tensor key, Scalar value) {
    stateActionMap.put(key, Scalars.intValueExact(value));
  }

  public Scalar getLogarithmicStateActionCount(Tensor state, Tensor action) {
    Tensor key = StateAction.key(state, action);
    return stateActionMap.containsKey(key) //
        ? LOGARITHMIC.apply(RealScalar.of(stateActionMap.get(key)))
        : DoubleScalar.NEGATIVE_INFINITY;
  }

  public DiscreteQsa inQsa(DiscreteModel discreteModel) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsa.assign(state, action, getLogarithmicStateActionCount(state, action));
    return qsa;
  }
}
