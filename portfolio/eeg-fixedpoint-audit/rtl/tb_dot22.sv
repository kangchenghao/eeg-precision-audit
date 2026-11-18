`timescale 1ns/1ps
module tb_dot22;
    parameter integer WB = 12;
    localparam integer AB = 16 + WB + 4;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 1;
    reg in_valid = 0;
    reg signed [15:0] x = 0;
    reg signed [WB-1:0] w = 0;
    wire out_valid;
    wire signed [AB-1:0] result;
    dot22 #(.WB(WB)) dut(.*);
    string stim;
    integer fd, rc, line_no, outputs;
    integer ri, vi, xi, wi, expected_valid;
    reg signed [63:0] expected_result;
    initial begin
        line_no = 0;
        outputs = 0;
        if (!$value$plusargs("STIM=%s", stim)) $fatal(1, "missing STIM");
        fd = $fopen(stim, "r");
        if (!fd) $fatal(1, "cannot open STIM");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%d %d %d %d %d %d\n",
                         ri, vi, xi, wi, expected_valid, expected_result);
            if (rc != 6) $fatal(1, "malformed stimulus row %0d rc=%0d", line_no, rc);
            @(negedge clk);
            rst_n = ri;
            in_valid = vi;
            x = xi;
            w = wi;
            // Reset is checked between active edges, not only at a clock edge.
            #1;
            if (!rst_n && (out_valid !== 1'b0 || result !== 0))
                $fatal(1, "asynchronous reset mismatch row %0d", line_no);
            @(posedge clk);
            #1;
            if (out_valid !== expected_valid[0])
                $fatal(1, "valid mismatch row %0d expected=%0d actual=%b", line_no, expected_valid, out_valid);
            if (expected_valid) begin
                outputs = outputs + 1;
                if ($signed(result) !== expected_result)
                    $fatal(1, "dot mismatch row %0d expected=%0d actual=%0d", line_no, expected_result, $signed(result));
            end
            line_no = line_no + 1;
        end
        $fclose(fd);
        if (!outputs) $fatal(1, "no completed dot products");
        $display("PASS dot22 WB=%0d rows=%0d outputs=%0d", WB, line_no, outputs);
        $finish;
    end
endmodule
