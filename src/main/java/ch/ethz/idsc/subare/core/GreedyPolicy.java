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
      Tensor states, //
      Tensor actions, //
      Tensor values, // <- values of states
      MoveInterface moveInterface) {
    Index statesIndex = Index.of(states);
    Index actionIndex = Index.of(actions);
    Map<Tensor, Tensor> map = new HashMap<>();
    for (Tensor state : states) {
      Tensor arg = Tensors.empty();
      for (int actionI = 0; actionI < actionIndex.size(); ++actionI) {
        Tensor next = moveInterface.move(state, actions.get(actionI));
        int nextI = statesIndex.indexOf(next);
        Scalar value = values.Get(nextI);
        arg.append(value);
      }
      int actionG = ArgMax.of(arg); // TODO use fair arg max
      map.put(state, actions.get(actionG));
    }
    return new GreedyPolicy(map);
  }

  final Map<Tensor, Tensor> map;

  private GreedyPolicy(Map<Tensor, Tensor> map) {
    this.map = map;
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    return KroneckerDelta.of(map.get(state), action);
  }

  public void print(Tensor states) {
    System.out.println("greedy:");
    for (Tensor state : states) {
      System.out.println(state + " -> " + map.get(state));
    }
  }
}
