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
  private final Map<Tensor, Integer> map = new HashMap<>();

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    map.put(key, map.containsKey(key) ? map.get(key) + 1 : 1);
  }

  @Override
  public Scalar counts(Tensor key) {
    return map.containsKey(key) ? RealScalar.of(map.get(key)) : RealScalar.ZERO;
  }

  @Override
  public boolean isEncountered(Tensor key) {
    return Sign.isPositive(counts(key));
  }

  public Scalar getLogarithmicCount(Tensor state, Tensor action) {
    Tensor key = StateAction.key(state, action);
    return map.containsKey(key) ? LOGARITHMIC.apply(RealScalar.of(map.get(key))) : RealScalar.of(Double.NEGATIVE_INFINITY);
  }

  public DiscreteQsa inQsa(DiscreteModel discreteModel) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsa.assign(state, action, getLogarithmicCount(state, action));
    return qsa;
  }
}
