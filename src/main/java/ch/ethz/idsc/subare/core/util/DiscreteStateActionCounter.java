// code by jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.ScalarUnaryOperator;
import ch.ethz.idsc.tensor.sca.Sign;

public class DiscreteStateActionCounter implements StateActionCounter, Serializable {
  private static final ScalarUnaryOperator LOGARITHMIC = scalar -> Log.of(scalar.add(RealScalar.ONE));
  // ---
  private final Map<Tensor, Integer> stateActionMap = new HashMap<>();
  private final Map<Tensor, Integer> stateMap = new HashMap<>();

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    Tensor state = stepInterface.prevState();
    stateMap.put(state, stateMap.containsKey(state) ? stateMap.get(state) + 1 : 1);
    stateActionMap.put(key, stateActionMap.containsKey(key) ? stateActionMap.get(key) + 1 : 1);
  }

  @Override
  public Scalar stateActionCount(Tensor key) {
    return stateActionMap.containsKey(key) ? RealScalar.of(stateActionMap.get(key)) : RealScalar.ZERO;
  }

  @Override
  public Scalar stateCount(Tensor state) {
    return stateMap.containsKey(state) ? RealScalar.of(stateMap.get(state)) : RealScalar.ZERO;
  }

  @Override
  public boolean isEncountered(Tensor key) {
    return Sign.isPositive(stateActionCount(key));
  }

  public Scalar getLogarithmicStateActionCount(Tensor state, Tensor action) {
    Tensor key = StateAction.key(state, action);
    return stateActionMap.containsKey(key) ? //
        LOGARITHMIC.apply(RealScalar.of(stateActionMap.get(key))) : RealScalar.of(Double.NEGATIVE_INFINITY);
  }

  public DiscreteQsa inQsa(DiscreteModel discreteModel) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsa.assign(state, action, getLogarithmicStateActionCount(state, action));
    return qsa;
  }
}
