// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.TerminalInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.ScalarUnaryOperator;

public class StateActionCounter extends DequeDigestAdapter implements DiscreteQsaSupplier {
  public static final ScalarUnaryOperator LOGARITHMIC = scalar -> Log.of(scalar.add(RealScalar.ONE));
  // ---
  private final DiscreteModel discreteModel;
  private final TerminalInterface terminalInterface;
  private final Map<Tensor, Integer> map = new HashMap<>();

  public StateActionCounter(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
    terminalInterface = discreteModel instanceof TerminalInterface ? (TerminalInterface) discreteModel : null;
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    for (StepInterface stepInterface : deque)
      increment(stepInterface.prevState(), stepInterface.action());
    // ---
    if (terminalInterface != null) { // count terminal state
      StepInterface stepInterface = deque.getLast();
      Tensor state1 = stepInterface.nextState();
      if (terminalInterface.isTerminal(state1)) {
        Tensor actions = discreteModel.actions(state1);
        if (actions.length() != 1)
          throw TensorRuntimeException.of(state1, actions);
        increment(state1, actions.get(0));
      }
    }
  }

  private void increment(Tensor state0, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state0, action);
    map.put(key, map.containsKey(key) ? map.get(key) + 1 : 1);
  }

  public Scalar getCount(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    return map.containsKey(key) ? RealScalar.of(map.get(key)) : RealScalar.ZERO;
  }

  @Override
  public DiscreteQsa qsa() {
    return qsa(s -> s);
  }

  public DiscreteQsa qsa(ScalarUnaryOperator scalarUnaryOperator) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsa.assign(state, action, scalarUnaryOperator.apply(getCount(state, action)));
    return qsa;
  }
}
