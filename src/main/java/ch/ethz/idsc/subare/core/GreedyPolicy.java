// code by jph
package ch.ethz.idsc.subare.core;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.ArgMax;
import ch.ethz.idsc.tensor.red.KroneckerDelta;

public class GreedyPolicy implements PolicyInterface {
  public static GreedyPolicy build( //
      StandardModel standardModel, //
      Index statesIndex, //
      Tensor values // <- values of states
  ) {
    Map<Tensor, Tensor> map = new HashMap<>();
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      final Tensor state = statesIndex.get(stateI);
      // bellman optimality equation:
      // v_*(s) == max_a Sum_{s',r} p(s',r | s,a) * (r + gamma * v_*(s'))
      // simplifies here to
      // v_*(s) == max_a (r + gamma * v_*(s'))
      Tensor va = Tensors.empty();
      // TODO can reduce code by streaming va
      Tensor actions = standardModel.actions(state);
      for (Tensor action : actions)
        va.append(standardModel.qsa(state, action, values));
      int argMax = ArgMax.of(va);
      map.put(state, actions.get(argMax));
    }
    return new GreedyPolicy(map);
  }

  private final Map<Tensor, Tensor> map;

  private GreedyPolicy(Map<Tensor, Tensor> map) {
    this.map = map;
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    return KroneckerDelta.of(map.get(state), action);
  }

  public Tensor bestFor(Tensor states) {
    return Tensor.of(states.flatten(0).map(key -> map.get(key)));
  }

  public void print(Tensor states) {
    System.out.println("greedy:");
    for (Tensor state : states) {
      System.out.println(state + " -> " + map.get(state));
    }
  }
}
