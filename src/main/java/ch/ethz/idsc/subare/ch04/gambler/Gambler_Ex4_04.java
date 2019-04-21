// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.IOException;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Put;

/* package */ enum Gambler_Ex4_04 {
  ;
  public static void main(String[] args) throws IOException {
    Gambler gambler = Gambler.createDefault();
    ValueIteration vi = new ValueIteration(gambler, gambler);
    Tensor record = Tensors.empty();
    for (int iters = 0; iters < 20; ++iters) {
      vi.step();
      record.append(vi.vs().values());
    }
    Tensor values = Last.of(record);
    // .untilBelow(RealScalar.of(1e-10));
    // System.out.println(values);
    Put.of(HomeDirectory.file("ex403_values"), values);
    Put.of(HomeDirectory.file("ex403_record"), record);
    PolicyBase policy = PolicyType.GREEDY.bestEquiprobable(gambler, vi.vs(), null);
    Policies.print(policy, gambler.states());
    Tensor greedy = Policies.flatten(policy, gambler.states());
    Put.of(HomeDirectory.file("ex403_greedy"), greedy);
  }
}
