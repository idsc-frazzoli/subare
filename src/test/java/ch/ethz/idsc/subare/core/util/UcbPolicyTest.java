// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.demo.airport.Airport;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import junit.framework.TestCase;

public class UcbPolicyTest extends TestCase {
  public void testSimple() {
    Airport airport = Airport.INSTANCE;
    DiscreteQsa qsa = DiscreteQsa.build(airport);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(airport, ConstantLearningRate.of(RealScalar.ZERO), //
        qsa, sac, PolicyType.EGREEDY.bestEquiprobable(airport, qsa, sac));
    for (Tensor state : airport.states()) {
      for (Tensor action : airport.actions(state)) {
        assertFalse(sarsa.sac().isEncountered(StateAction.key(state, action)));
        assertTrue(sarsa.sac().stateActionCount(StateAction.key(state, action)).equals(RealScalar.ZERO));
        assertTrue(sarsa.sac().stateCount(state).equals(RealScalar.ZERO));
      }
    }
    Tensor state = airport.states().get(0);
    Tensor action = airport.actions(state).get(0);
    Tensor nextState = airport.move(state, action);
    sarsa.digest(new StepAdapter(state, action, RealScalar.ZERO, nextState));
    for (Tensor s : airport.states()) {
      for (Tensor a : airport.actions(state)) {
        if (state.equals(s)) {
          assertTrue(sarsa.sac().stateCount(s).equals(RealScalar.ONE));
          if (action.equals(a)) {
            assertTrue(sarsa.sac().isEncountered(StateAction.key(s, a)));
            assertTrue(sarsa.sac().stateActionCount(StateAction.key(s, a)).equals(RealScalar.ONE));
          } else {
            assertTrue(sarsa.sac().stateActionCount(StateAction.key(s, a)).equals(RealScalar.ZERO));
            assertFalse(sarsa.sac().isEncountered(StateAction.key(s, a)));
          }
        } else {
          assertTrue(sarsa.sac().stateCount(s).equals(RealScalar.ZERO));
        }
      }
    }
  }

  public void testUcb() {
    Airport airport = Airport.INSTANCE;
    DiscreteQsa qsa = DiscreteQsa.build(airport);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(airport, ConstantLearningRate.of(RealScalar.ZERO), //
        qsa, sac, PolicyType.EGREEDY.bestEquiprobable(airport, qsa, sac));
    DiscreteQsa ucbInQsa = UcbUtils.getUcbInQsa(airport, qsa, sarsa.sac());
    for (Tensor state : airport.states()) {
      for (Tensor action : airport.actions(state)) {
        assertTrue(UcbUtils.getUpperConfidenceBound(state, action, qsa.value(state, action), sarsa.sac(), airport).equals(DoubleScalar.POSITIVE_INFINITY));
        assertTrue(ucbInQsa.value(state, action).equals(DoubleScalar.POSITIVE_INFINITY));
      }
    }
    Tensor state = airport.states().get(0);
    Tensor action = airport.actions(state).get(0);
    Tensor nextState = airport.move(state, action);
    sarsa.digest(new StepAdapter(state, action, RealScalar.ZERO, nextState));
    ucbInQsa = UcbUtils.getUcbInQsa(airport, qsa, sarsa.sac());
    for (Tensor s : airport.states()) {
      for (Tensor a : airport.actions(s)) {
        if (s.equals(state) && a.equals(action))
          assertTrue(ucbInQsa.value(s, a).equals(RealScalar.ZERO));
        else
          assertTrue(ucbInQsa.value(s, a).equals(DoubleScalar.POSITIVE_INFINITY));
      }
    }
  }
}
