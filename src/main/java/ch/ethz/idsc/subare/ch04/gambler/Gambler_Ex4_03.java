// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.subare.core.GreedyPolicy;
import ch.ethz.idsc.subare.core.ValueFunctions;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Put;

class Gambler_Ex4_03 {
  public static void main(String[] args) throws IOException {
    Gambler ga = new Gambler(RealScalar.of(0.4));
    Tensor values = ValueFunctions.bellmanIterationMax(ga, ga.statesIndex, RealScalar.ONE, RealScalar.of(1e-9));
    Put.of(new File("/home/datahaki/ex403_values"), values);
    GreedyPolicy greedyPolicy = GreedyPolicy.build(ga, ga.statesIndex, values);
    Tensor greedy = greedyPolicy.bestFor(ga.states);
    Put.of(new File("/home/datahaki/ex403_greedy"), greedy);
    greedyPolicy.print(ga.states);
  }
}
