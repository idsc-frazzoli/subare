// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public class StateActionCounter extends DequeDigestAdapter implements DiscreteQsaSupplier {
  private final DiscreteModel discreteModel;
  private final Map<Tensor, Integer> map = new HashMap<>();

  public StateActionCounter(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    for (StepInterface stepInterface : deque) {
      Tensor state0 = stepInterface.prevState();
      Tensor action = stepInterface.action();
      Tensor key = DiscreteQsa.createKey(state0, action);
      map.put(key, map.containsKey(key) ? map.get(key) + 1 : 1);
    }
  }

  public Scalar getCount(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    return map.containsKey(key) ? RealScalar.of(map.get(key)) : RealScalar.ZERO;
  }

  @Override
  public DiscreteQsa qsa() {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsa.assign(state, action, getCount(state, action));
    return qsa;
  }
}
